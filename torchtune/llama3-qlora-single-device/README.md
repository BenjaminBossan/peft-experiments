# Comparison of torchtune and PEFT results for Llama3 8b instruct LoRA-finetuning on single device

## Setup

- PEFT commit hash: 1b16753a6aca2fc9581823c8e2efd628b7d2b676
- torchtune commit hash: ca1d7a1584f573b2dde719c4c9a0f678bff76089
- transformers commit hash: 7f552e28e0aca00ce60868c7620f7463eab60e14
- bitsandbytes
  - v0.43.3
  - `load_in_4bit=True`
  - `bnb_4bit_compute_dtype=torch.bfloat16`
  - `bnb_4bit_use_double_quant=True`
  - `bnb_4bit_quant_type="nf4"`
- torch v2.3.1
- torchtune config: `bb_8B_qlora_single_device.yaml`
- no compilation
- no quantization
- dtype: bfloat16 (also for LoRA weights)
- Single device training
  - NVIDIA GeForce RTX 4090
  - Build cuda_12.0.r12.0/compiler.32267302_0

## Training

- PEFT: See `train-peft.ipynb`
- torchtune: `tune run lora_finetune_single_device --config llama3/bb_8B_qlora_single_device.yaml`
  - Note that activation checkpointing has been disabled for torchtune. This is because it was not trivial to add it to PEFT/transformers (as I want to avoid using `Trainer`), so to keep things equal, it was disabled for both.

## Results

- PEFT logs: `log_peft.txt`
- torchtune logs: `log_torchtune.txt`
- Comparison notebook: `compare-logs.ipynb`

## Conclusion

- Losses are very closely matched
- Memory: torchtune slightly lower (~19GiB) than PEFT + bitsandbytes (~20GiB)
- Tokens per second: torchtune much lower (~670) than PEFT + bitsandbytes (~2300), not sure why, probably some optimization in transformers (cache?)

Note that torchtune uses torchao NF4 for quantization (weight type is `torchao.dtypes.nf4tensor.NF4Tensor`), whereas PEFT uses bitsandbytes.
