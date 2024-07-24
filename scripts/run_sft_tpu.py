# export HUGGING_FACE_HUB_TOKEN=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
# pip install transformers
# python run_sft_tpu.py ../recipes/custom/config.yaml
# ~/.cache/huggingface/hub
# dataset% rm -r *
# alignment-handbook % python scripts/run_sft_tpu.py recipes/custom/config.yaml
import logging
import random
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import setup_chat_format, SFTTrainer, SFTConfig

from alignment import H4ArgumentParser, ModelArguments, DataArguments, get_datasets, get_tokenizer, \
    apply_chat_template, decontaminate_humaneval, get_peft_config

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    # print(column_names)
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, use_cache=False)
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    # print(len(raw_datasets["train"]))
    # print(list(raw_datasets["train"].features))
    # print(raw_datasets["train"]["text"][0])
    # print(raw_datasets.keys())

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1)
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples / num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # print("After Decontamination")
    # print(train_dataset[0])

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
            # print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    # print(training_args)
    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs
    )


if __name__ == "__main__":
    main()
