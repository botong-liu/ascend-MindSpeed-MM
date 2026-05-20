# 完整调用链（以 pretrain_transformers.py 为入口）

## 启动阶段

```text
pretrain_transformers.py (入口)
  └── pretrain()                            ← mindspeed_mm/training.py:86
      ├── initialize_megatron()             ← 初始化分布式环境
      ├── model_provider()                  ← pretrain_transformers.py:58
      │   └── TransformersModel(config)     ← mindspeed_mm/models/transformers_model.py:34
      │       ├── ModelHub.build()          ← 构建 model class
      │       └── model_cls.from_pretrained()  ← 加载 Qwen3VLMoeForConditionalGeneration
      ├── build_train_valid_test_data_iterators()
      └── train()                           ← mindspeed_mm/training.py:346
          └── train_step()                  ← mindspeed_mm/training.py:736
              ├── get_forward_backward_func()
              └── forward_backward_func()   ← Megatron core
```

## 前向训练（forward_backward_no_pipelining）

对于非 PP（pipeline_model_parallel_size=1）的情况，Megatron 使用 `forward_backward_no_pipelining`：

```text
forward_backward_no_pipelining()
  └── forward_step_func(data_iterator, model)  ← pretrain_transformers.py:127
      ├── model(**batch_data)                  ← TransformersModel.forward
      │   └── self.model(**batch_data)         ← Qwen3VLMoeForConditionalGeneration.forward
      │       └── self.model()                 ← Qwen3VLMoeModel.forward
      │           └── self.language_model()    ← Qwen3VLMoeTextModel.forward
      │               └── decoder_layer iter...← Qwen3VLMoeTextDecoderLayer.forward (N层循环)
      │                   ├── self_attn()      ← Qwen3VLMoeTextAttention.forward
      │                   └── self.mlp()       ← MoE层 或 Dense层
      └── loss_func(output_tensor)             ← pretrain_transformers.py:95
  └── loss.backward()                          ← Megatron core 内部触发
```

## MoE 层前向细分类

当启用 EP (Expert Parallel) 时，MoE 层走 `models/transformers/` 路径：

```text
self.mlp = Qwen3VLMoeTextSparseMoeBlock (MoE层)
  └── Qwen3VLMoeTextSparseMoeBlock.forward  ← fsdp/models/.../modeling_qwen3_vl_moe.py:191
      ├── self.gate() → Qwen3VLMoeTextRouter → router_logits + top-k routing
      └── self.experts() → Qwen3VLMoeTextExperts.forward
          (非EP路径: npu_moe_token_permute + npu_group_gemm + unpermute)
```

当启用 EP 时，forward 被替换为：

```text
Qwen3VLNpuFusedMoETextExperts.ep_forward      ← models/transformers/.../modeling_qwen3_vl_moe.py:144
  └── fused_ep_forward()                      ← mindspeed_mm/models/common/fused_moe.py:26
      ├── dispatch_preprocess()               ← 统计token，构造input/output_splits
      ├── alltoall_dispatch()                 ← permute + all_to_all (派发)
      │   └── all_to_all_EP()                 ← communications.py
      │       └── _AllToAll.forward
      │           └── _ep_all_to_all()        ← 实际 all_to_all_single
      ├── npu_group_gemm + swiglu             ← Expert 本地计算
      └── alltoall_combine()                  ← all_to_all (回传) + unpermute
          └── all_to_all_EP()
              └── _ep_all_to_all()
```

## 反向传播

```text
loss.backward()   ← Megatron core 在 forward_backward_no_pipelining 内部调用
  ├── lm_head.weight 梯度
  └── decoder layers 反向 (L-1 → 0 逆向遍历)
      ├── mlp 反向
      │   └── _AllToAll.backward              ← communications.py:212
      │       └── _ep_all_to_all()            ← 交换 scatter/gather dims
      │           └── all_to_all_single()     ← 反向EP通信
      └── attention 反向
  └── embed_tokens 梯度
```

## 关键文件对应关系

| 文件 | 角色 | 关键函数/类 |
|---|---|---|
| `pretrain_transformers.py` | 训练入口 | `model_provider`, `forward_step`, `loss_func` |
| `mindspeed_mm/training.py` | Megatron 训练调度 | `pretrain() → train() → train_step()` |
| `mindspeed_mm/models/transformers_model.py` | 模型封装 | `TransformersModel.forward` |
| `mindspeed_mm/fsdp/models/qwen3vl/modeling_qwen3_vl_moe.py` | FSDP模型定义 | `Qwen3VLMoeForConditionalGeneration`, `Qwen3VLMoeTextDecoderLayer`, `Qwen3VLMoeTextSparseMoeBlock` |
| `mindspeed_mm/models/transformers/qwen3vl/modeling_qwen3_vl_moe.py` | Megatron EP路径 | `Qwen3VLNpuFusedMoETextExperts.ep_forward` |
| `mindspeed_mm/models/common/fused_moe.py` | EP核心流程 | `fused_ep_forward`, `dispatch_preprocess`, `alltoall_dispatch`, `alltoall_combine` |
| `mindspeed_mm/models/common/communications.py` | 通信原语 | `_ep_all_to_all`, `_AllToAll (forward+backward)` |

## 执行顺序总结

```text
pretrain_transformers.py:203-213
  → mindspeed_mm/training.py:86 (pretrain)
    → mindspeed_mm/training.py:346 (train loop, iteration循环)
      → mindspeed_mm/training.py:736 (train_step, micro-batch循环)
        → Megatron forward_backward_no_pipelining (前向+反向核心)
          → pretrain_transformers.py:127 (forward_step)
            → TransformersModel.forward → Qwen3VLMoeForConditionalGeneration.forward
              → 逐层 Qwen3VLMoeTextDecoderLayer.forward (FSDP版)
                → attention (Qwen3VLMoeTextAttention)
                → MoE mlp (Qwen3VLMoeTextSparseMoeBlock → 或 EP替换路径 fused_ep_forward)
                  → dispatch_preprocess → alltoall_dispatch → _ep_all_to_all
                  → Expert本地计算 (npu_group_gemm)
                  → alltoall_combine → _ep_all_to_all
            → loss_func (计算loss)
          → loss.backward() (自动微分, 反向遍历)
            → _AllToAll.backward → _ep_all_to_all (EP反向通信)
        → optimizer.step()
```
