# Using trl Trainer and accelerate (trl-0.9.6, accelerate-0.33.0)
# accelerate launch --config_file fsdp_config.yaml train-hf.py
#
# for DoRA:
# USE_DORA=1 accelerate launch --config_file fsdp_config.yaml train-hf.py

import json
import os
import random
import time
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from torchtune.datasets import InstructDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def get_memory_stats(device: torch.device, reset_stats: bool = True) -> dict:
    # torchtune: log memory stats
    if device.type != "cuda":
        raise ValueError(
            f"Logging memory stats is only supported on CUDA devices, got {device}"
        )

    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)

    memory_stats = {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_mem_alloc,
        "peak_memory_reserved": peak_mem_reserved,
    }
    return memory_stats


def my_padded_collate(
    batch,
    padding_idx,
    ignore_idx=-100,
):
    # same as torchtune but tokens is renamed to input_ids
    input_ids = pad_sequence(
        [torch.tensor(x["input_ids"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )

    return {"input_ids": input_ids, "labels": labels}


class MyInstructDataset(InstructDataset):
    def _prepare_sample(self, *args, **kwargs):
        # same as InstructDataset but rename tokens to input_ds
        sample = super()._prepare_sample(*args, **kwargs)
        sample["input_ids"] = sample.pop("tokens")
        return sample



def truncate(
    tokens,
    max_seq_len,
    eos_id,
):
    # same as torchtune.utils.truncate
    tokens_truncated = tokens[:max_seq_len]
    if eos_id is not None and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated



class TokenizerWrapper:
    # wrap transformers tokenizer but add torchtune method
    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def tokenize_messages(self, messages, max_seq_len=None, add_eos=True):
        tokens = [self.hf_tokenizer.bos_token_id]
        mask = [True]
        for message in messages:
            tokenized_message = self.hf_tokenizer.encode(
                message.text_content, add_special_tokens=False
            )

            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if max_seq_len and len(tokens) >= max_seq_len:
                break

        if add_eos:
            tokens = tokens + [self.hf_tokenizer.eos_token_id]
            mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.hf_tokenizer.eos_token_id)
            mask = truncate(mask, max_seq_len, True)

        return tokens, mask

    def __getattr__(self, name):
        return getattr(self.hf_tokenizer, name)


class SFTTrainerWithLogs(SFTTrainer):
    # log train loss, tokens per step, and learning rate
    # also log tokens per second, memory allocated, and memory reserved

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tokens = 0
        self.duration = 0.0
        self.grad_accum_step = 0
        self.extra_logs = {}

    def get_train_dataloader(self, *args, **kwargs):
        shuffle = True
        ds = self.train_dataset
        sampler = torch.utils.data.DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        packed = False
        dataloader = torch.utils.data.DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=(
                partial(
                    my_padded_collate,
                    padding_idx=1,  # tokenizers pad_token_id
                    ignore_idx=-100,
                )
                if not packed
                else None
            ),
        )
        return self.accelerator.prepare(dataloader)

    def training_step(self, model, inputs):
        tic = time.perf_counter()
        output = super().training_step(model, inputs)
        toc = time.perf_counter()
        self.duration += toc - tic
        num_tokens = inputs["input_ids"].numel()
        self.num_tokens += num_tokens

        if self.accelerator:
            print = self.accelerator.print
        else:
            print = print

        # check if logging is needed, consider steps and grad accumulation
        # there is probably a more elegant way to do this
        self.grad_accum_step += 1
        do_log = False
        global_step, logging_steps = self.state.global_step, self.args.logging_steps
        do_log = (global_step % logging_steps == 0) and (global_step % self.args.gradient_accumulation_steps == 0)

        if do_log:
            tokens_per_sec = self.num_tokens / self.duration
            memo = get_memory_stats(self.model.device)
            out_dict = {
                "tokens_per_second": tokens_per_sec,
                "peak_memory_active": memo["peak_memory_active"],
                "peak_memory_alloc": memo["peak_memory_alloc"],
                "peak_memory_reserved": memo["peak_memory_reserved"],
            }
            self.extra_logs.update(out_dict)
            self.num_tokens = 0
            self.duration = 0.0

        return output

    def log(self, logs: dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}

        output.update(self.extra_logs)
        output["step"] = self.state.global_step
        with open(log_file, "a") as f:
            f.write(json.dumps(output) + "\n")

        self.extra_logs.clear()
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


def train():
    #########
    # SETUP #
    #########

    torch.cuda.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer = TokenizerWrapper(tokenizer)
    ds = load_dataset(ds_name)


    from torchtune.datasets._instruct import _get_component_from_path
    template = _get_component_from_path("torchtune.data.AlpacaInstructTemplate")
    ds = MyInstructDataset(
        tokenizer, ds_name, train_on_input=True, max_seq_len=max_seq_len, split="train", template=template,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        use_dora=use_dora,
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=total_epochs,
        save_strategy="no",
        bf16=True,
        learning_rate=lr,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=num_warmup_steps,
        dataset_text_field="input",
        max_seq_length=max_seq_len,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=5,
        remove_unused_columns=False,
    )

    trainer = SFTTrainerWithLogs(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        peft_config=lora_config,
        packing=False,
    )
    trainer.train()


if __name__ == "__main__":
    ##############
    # PARAMETERS #
    ##############

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype = torch.bfloat16
    weight_decay = 0.01
    lr = 0.0003
    gradient_accumulation_steps = 4
    num_warmup_steps = 100

    # data
    ds_name = "yahma/alpaca-cleaned"
    max_seq_len = 512
    total_epochs = 1
    batch_size = 2
    logging_steps = 4
    output_dir = "/tmp/peft/sft"
    randint = random.randint(0, 100000)
    log_file = f"{output_dir}/log_{randint}.txt"
    print(f"logs are written to {log_file}")

    # LoRA
    rank = 8
    alpha = 16
    target_modules = ["q_proj", "v_proj"]
    dropout = 0.05
    use_dora = bool(int(os.environ.get("USE_DORA", "0")))
    if use_dora:
        print("USING DORA")

    train()
