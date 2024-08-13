# Comparison of torchtune and PEFT results for Llama3 8b instruct LoRA-finetuning with FSDP on two devices

## Setup

- PEFT commit hash: 1b16753a6aca2fc9581823c8e2efd628b7d2b676
- torchtune commit hash: ca1d7a1584f573b2dde719c4c9a0f678bff76089
- transformers commit hash: 7f552e28e0aca00ce60868c7620f7463eab60e14
- torch v2.3.1
- torchtune config: `bb_8B_lora.yaml`
- no compilation
- no quantization
- dtype: bfloat16 (also for LoRA weights)
- TWo device training
  - NVIDIA GeForce RTX 4090
  - Build cuda_12.0.r12.0/compiler.32267302_0

## Training

Assuming 2 GPUs:

- PEFT: `torchrun --nproc_per_node 2 train.py`
- torchtune: `tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/bb_8B_lora.yaml`

Note that as expected, with only 2 devices, training with FSDP is actually much slower than single device training.

## Results

- PEFT logs: `log_peft.txt`
- torchtune logs: `log_torchtune.txt`
- Comparison notebook: `compare-logs.ipynb`

## Conclusion

- Losses are close but not matched, unlike with single device training.
- Memory is a little bit lower for torchtune (maybe more efficient autowrap policy?).
- Tokens per second are also very close.

# Using Trainer and accelerate

## Setup

- Same as above
- accelerate: v0.33.0
- transformers: e234061cddd28bb8b82144833241883816289e40
- trl: v0.9.6

## Training

- `accelerate launch --config_file fsdp_config.yaml train-hf.py`
- For DoRA, add `USE_DORA=1` env var
