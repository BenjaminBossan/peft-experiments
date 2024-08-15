# Comparison of answer.ai fsdp_qlora and PEFT QLoRA results for Llama2 7b LoRA-finetuning with FSDP on two devices

## Setup

- PEFT commit hash: 22f042a107b036c2894482badb26ea9eb3567b7a
- fsdp_qlora repo hash: cec4386685702a9f6f36fbb04a3a9a370f0981eb
- transformers commit hash: e4522fe399641add9b1f3207923752a35ea7fdbb
- torch v2.3.1
- bitsandbytes v0.43.3
- no compilation
- no quantization
- used bitsandbytes, not HQQ
- dtype: bfloat16 (also for LoRA weights)
- Two device training
  - NVIDIA GeForce RTX 4090
  - Build cuda_12.0.r12.0/compiler.32267302_0

## Training

Use the included `train.py` script from this directory. It is adapted from the `train.py` file from the [fsdp_qlora repo](https://github.com/AnswerDotAI/fsdp_qlora/blob/cec4386685702a9f6f36fbb04a3a9a370f0981eb/train.py). The main changes are:

- for PEFT, use `BitsAndBytesConfig`
- for PEFT, add DoRA option
- add extra logging to files

The script should be called like so:

- PEFT: `python train.py --model_name meta-llama/Llama-2-7b-hf --train_type qlora --batch_size 1 --context_length 512 --precision bf16 --use_gradient_checkpointing false --use_cpu_offload true --dataset alpaca_sample --low_memory false --reentrant_checkpointing true --lora_rank 8 --lora_alpha 16 --verbose true --gradient_accumulation_steps 4 --train_type qlora`
- Answer.ai: `python train.py --model_name meta-llama/Llama-2-7b-hf --train_type qlora --batch_size 1 --context_length 512 --precision bf16 --use_gradient_checkpointing false --use_cpu_offload true --dataset alpaca_sample --low_memory false --reentrant_checkpointing true --lora_rank 8 --lora_alpha 16 --verbose true --gradient_accumulation_steps 4 --train_type custom_qlora`

## Results

- PEFT logs: `log_peft.txt`
- answer.ai logs: `log_answerai.txt`
- Comparison notebook: `compare-logs.ipynb`

## Conclusion

- Losses are matched exactly
- Memory usage is very close
- Answer.ai is ~20% faster for unclear reasons.
