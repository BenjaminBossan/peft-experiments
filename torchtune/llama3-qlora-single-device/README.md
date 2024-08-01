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

## Results

- PEFT logs: `log_peft_bnb-0.43.3.txt`
- torchtune logs: `log_torchtune.txt`
- Comparison notebook: `compare-logs.ipynb`

## Conclusion

- Losses are very closely matched
- Memory: torchtune much lower (<8GiB) than PEFT + bitsandbytes (~21GiB)
- Tokens per second: torchtune much lower (465) than PEFT + bitsandbytes (1410)

It looks like torchtune uses torchao for quantization (weight type is `torchao.dtypes.nf4tensor.NF4Tensor`), which leads to very different memory and performance profiles than bitsandbytes. The extra memory required by PEFT + bnb can be traced back to the activations -- when choosing batch size of 1 and sequence length of 32, memory comes down to ~8GiB. Further investigation required.