# Comparison of torchtune and PEFT results for Llama3 8b instruct LoRA-finetuning on single device

## Setup

- PEFT commit hash: 1b16753a6aca2fc9581823c8e2efd628b7d2b676
- torchtune commit hash: ca1d7a1584f573b2dde719c4c9a0f678bff76089
- transformers commit hash: 7f552e28e0aca00ce60868c7620f7463eab60e14
- torch v2.3.1
- torchtune config: `bb_8B_lora_single_device.yaml`
- no compilation
- no quantization
- dtype: bfloat16 (also for LoRA weights)
- Single device training
  - NVIDIA GeForce RTX 4090
  - Build cuda_12.0.r12.0/compiler.32267302_0

## Training

- PEFT: See `train-peft.ipynb`
- torchtune: `tune run lora_finetune_single_device --config llama3/bb_8B_lora_single_device.yaml`

## Results

- PEFT logs: `log_peft.txt`
- torchtune logs: `log_torchtune.txt`
- Comparison notebook: `compare-logs.ipynb`

## Conclusion

- Results are very closely matched, MSE between losses is 1.8e-5.
- Tokens per second are also close, with PEFT being ~20% faster. Not sure where the difference comes from. I thought it might be because checkpointing was enabled when training the torchtune model, but even when disabling it, the times are pretty much the same.
