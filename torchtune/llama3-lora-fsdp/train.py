# torchrun --nproc_per_node <N> train.py
#
# for DoRA:
# USE_DORA=1 torchrun --nproc_per_node <N> train.py

import os
import time
from functools import partial

import torch
from peft import LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from torch import nn
from torch.distributed import destroy_process_group, get_rank, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from torchtune.modules import get_cosine_schedule_with_warmup
from torchtune.datasets import InstructDataset
from torchtune.datasets._instruct import _get_component_from_path
from torchtune.utils import padded_collate, get_memory_stats
from torchtune.utils.metric_logging import DiskLogger
from torchtune.models.llama3 import llama3_tokenizer


def train(metric_logger):
    #########
    # SETUP #
    #########

    torch.cuda.manual_seed(0)
    base_path = os.path.expanduser("~/work/clones/torchtune/recipes/configs")
    tokenizer = llama3_tokenizer(os.path.join(base_path, "Meta-Llama-3-8B-Instruct/original/tokenizer.model"))
    ds = InstructDataset(
        tokenizer, ds_name, train_on_input=True, max_seq_len=max_seq_len, split="train", template=template,
    )
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
                padded_collate,
                padding_idx=128004,
                ignore_idx=-100,
            )
            if not packed
            else None
        ),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False) # torchtune uses bf16
    if is_rank_zero:
        print("="*20)
        print(model)
        print({p.dtype for p in model.parameters()})
        model.print_trainable_parameters()
        print("="*20)

    model = FSDP(
        module=model,
        #auto_wrap_policy=auto_wrap_policy,
        auto_wrap_policy=fsdp_auto_wrap_policy(model),
        sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        device_id=device,
        # this recipe does not currently support mixed precision training
        mixed_precision=None,
        # Ensure we broadcast params and buffers from rank 0
        sync_module_states=True,  # TODO: False?
        # Initialize empty modules on all non-zero ranks
        param_init_fn=(
            lambda module: (
                module.to_empty(device=torch.device("cuda"), recurse=False)
                if not is_rank_zero
                else None
            )
        ),
    )

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    num_training_steps = total_epochs * steps_per_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if is_rank_zero:
        print(f"{total_epochs=}, {steps_per_epoch=}, {num_training_steps=}")

    ##############
    # TRAIN LOOP #
    ##############

    t0 = time.perf_counter()
    running_loss = 0
    num_tokens = 0
    epochs_run = 0
    global_step = 0
    log_peak_memory_stats = True

    # epochs_run should be non-zero when we're resuming from a checkpoint
    for curr_epoch in range(epochs_run, total_epochs):
        # Update the sampler to ensure data is correctly shuffled across epochs
        # in case shuffle is True
        sampler.set_epoch(curr_epoch)

        pbar = tqdm(total=steps_per_epoch, disable=not is_rank_zero)
        for idx, batch in enumerate(dataloader):
            if (
                max_steps_per_epoch is not None
                and (idx // gradient_accumulation_steps)
                == max_steps_per_epoch
            ):
                break

            # Both are shape [b, s]
            tokens, labels = batch["tokens"], batch["labels"]
            # Get the attention mask and position ids from the dataset if they
            # exist. Currently, only sample packing in PackedDataset returns these
            mask = batch.get("mask", None)  # shape [b, s, s]
            input_pos = batch.get("input_pos", None)  # shape [b, s]

            tokens = tokens.to(device)
            num_tokens += tokens.numel()
            labels = labels.to(device)
            mask = mask.to(device) if mask is not None else None
            input_pos = (
                input_pos.to(device) if input_pos is not None else None
            )

            # bb logits = model(tokens, mask=mask, input_pos=input_pos)
            logits = model(tokens).logits
            # Shift so that tokens < n predict n
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)
            # Compute loss
            loss = loss_fn(logits, labels)
            # free logits otherwise it peaks backward memory
            del logits

            loss = loss / gradient_accumulation_steps
            running_loss += loss
            loss.backward()

            # Step with optimizer
            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                # Update the number of steps when the weights are updated
                global_step += 1

                loss_to_log = running_loss.item()
                pbar.update(1)
                pbar.set_description(
                    f"{curr_epoch + 1}|{global_step}|Loss: {loss_to_log}"
                )

                # Log per-step metrics
                if global_step % log_every_n_steps == 0:
                    time_per_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        "lr": optimizer.param_groups[0]["lr"],
                        "tokens_per_second_per_gpu": num_tokens / time_per_step,
                    }
                    if (
                        device.type == "cuda"
                        and log_peak_memory_stats
                    ):
                        log_dict.update(
                            get_memory_stats(device=device)
                        )
                    if is_rank_zero:
                        metric_logger.log_dict(
                            log_dict,
                            step=global_step,
                        )

                # Reset running stats for the next step
                running_loss = 0
                num_tokens = 0
                t0 = time.perf_counter()

        epochs_run += 1


def main():
    if is_rank_zero:
        metric_logger = DiskLogger("/tmp/peft/llama3-8b-lora-fsdp")
    else:
        metric_logger = None

    try:
        train(metric_logger)
    finally:
        if is_rank_zero:
            metric_logger.close()
        destroy_process_group()


if __name__ == "__main__":
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    init_process_group("nccl")
    rank = get_rank()
    is_rank_zero = rank == 0

    ##############
    # PARAMETERS #
    ##############

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype = torch.bfloat16
    device = torch.device(rank)
    weight_decay = 0.01
    lr = 0.0003
    gradient_accumulation_steps = 4
    log_every_n_steps = 1
    num_warmup_steps = 100

    # data
    ds_name = "yahma/alpaca-cleaned"
    template = _get_component_from_path("torchtune.data.AlpacaInstructTemplate")
    shuffle = True
    max_seq_len = 512
    total_epochs = 1
    batch_size = 2
    max_steps_per_epoch = None

    # LoRA
    lora_rank = 8
    lora_alpha = 16
    target_modules = ["q_proj", "v_proj"]
    dropout = 0.05
    use_dora = bool(int(os.environ.get("USE_DORA", "0")))
    if use_dora and is_rank_zero:
        print("USING DORA")

    main()
