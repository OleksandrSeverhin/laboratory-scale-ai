#!/usr/bin/env python3
import time

import numpy as np
import torch
import bitsandbytes as bnb
import logging
import sys
import transformers
import datasets
import argparse
import wandb
import pandas as pd

from datasets import load_dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from os import path, mkdir, getenv, makedirs
from typing import Mapping
from sklearn.model_selection import train_test_split

from finetune_functions import get_model_and_tokenizer, get_lora_model, get_default_trainer, get_dataset_slices
from evaluate_em import evaluate_hf_model_em, MODEL_SUFFIXES, system_message, transaction, examples


def format_data_as_instructions(data: Mapping,
                                tokenizer: AutoTokenizer,
                                system_message: str = '###',
                                transaction: str = '###',
                                nshots=0) -> list[str]:
    """
    Formats text data as instructions for the model. Can be used as a formatting function for the trainer class.
    """

    output_texts = []

    # Iterate over the test set
    for idx in range(len(data['prompt'])):
        question = data['prompt'][idx]

        chat = [
            {"role": "user", "content": system_message}
        ]
        for shot in range(1, nshots + 1):
            if chat[-1]['role'] == 'user':
                chat[-1]['content'] += examples[f'example_{shot}_question']
            else:
                chat.append({"role": "user", "content": examples[f'example_{shot}_question']})
            chat.append({"role": "assistant", "content": examples[f'example_{shot}_response']})
        if chat[-1]['role'] == 'user':
            chat[-1]['content'] += question + transaction
        else:
            chat.append({"role": "user", "content": question + transaction})
        chat.append({"role": "assistant", "content": data['overall_label'][idx]})

        output_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

        if 'falcon' in tokenizer.name_or_path:
            output_text = output_text.replace('user', 'User:')
            output_text = output_text.replace('assistant', 'Assistant:')
            output_text = output_text.replace('Assistant:!', 'assistant!')
            output_text = output_text.replace('Assistant:,', 'assistant,')
            output_text = output_text.replace('<|im_start|>', '')
            output_text = output_text.replace('<|im_end|>', '')

        output_texts.append(output_text)

    return output_texts

if __name__ == '__main__':
    hf_login()
    
    # Load the gpqa_diamond dataset from Hugging Face
    df = pd.read_excel('gpqa_diamond_15_custom.xlsx', engine="openpyxl")

    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')

    # Model ID
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-1B', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='False', help='Whether to resume from a checkpoint.')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')
    parser.add_argument('--use_mps_device', type=str, default='False', help='Whether to use an MPS device.')
    parser.add_argument('--max_memory', type=str, default='12000MB', help='The maximum memory per GPU, in MB.')

    # Model arguments
    parser.add_argument('--gradient_checkpointing', type=str, default='True', help='Whether to use gradient checkpointing.')
    parser.add_argument('--quantization_type', type=str, default='4bit', help='The quantization type to use for fine-tuning.')
    parser.add_argument('--lora', type=str, default='True', help='Whether to use LoRA.')
    parser.add_argument('--tune_modules', type=str, default='linear4bit', help='The modules to tune using LoRA.')
    parser.add_argument('--exclude_names', type=str, default='lm_head', help='The names of the modules to exclude from tuning.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default=df, help='The dataset to use for fine-tuning.')
    parser.add_argument('--version', type=str, default='3.0.0', nargs='?', help='The version of the dataset to use for fine-tuning.')
    parser.add_argument('--input_col', type=str, default='article', help='The name of the input column in the dataset.')
    parser.add_argument('--target_col', type=str, default='highlights', help='The name of the target column in the dataset.')
    parser.add_argument('--train_slice', type=str, default='train[:50]', help='The slice of the training dataset to use for fine-tuning.')
    parser.add_argument('--validation_slice', type=str, default='validation[:10]', help='The slice of the validation dataset to use for fine-tuning.')
    parser.add_argument('--test_slice', type=str, default='test[:10]', help='The slice of the test dataset to use for fine-tuning.')

    # Saving arguments
    parser.add_argument('--save_model', type=str, default='True', help='Whether to save the fine-tuned model and tokenizer.')
    parser.add_argument('--save_dir', type=str, default='finetuned_model', help='The directory to save the fine-tuned model and tokenizer.')
    parser.add_argument('--peft_save_dir', type=str, default='peft_model', help='The directory to save the PEFT model.')

    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs', help='The directory to save the log file.')
    parser.add_argument('--log_level', type=str, default='info', help='The log level to use for fine-tuning.')
    parser.add_argument('--logging_first_step', type=str, default='True', help='Whether to log the first step.')
    parser.add_argument('--logging_steps', type=int, default=1, help='The number of steps between logging.')
    parser.add_argument('--run_name', type=str, default='peft_finetune', help='The name of the run, for logging.')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='True', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='peft_finetune', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Prompt arguments
    parser.add_argument('--start_prompt', type=str, default=None, help='The start prompt to add to the beginning of the input text.')
    parser.add_argument('--end_prompt', type=str, default=' Label? [/INST]', help='The end prompt to add to the end of the input text.')
    parser.add_argument('--suffix', type=str, default='</s>', help='The suffix to add to the end of the input and target text.')
    parser.add_argument('--max_seq_length', type=int, default=974, help='The maximum sequence length to use for fine-tuning.')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='mistral',
                        help='Whether to use the default prompts for a model')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for fine-tuning.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='The number of gradient accumulation steps to use for fine-tuning.')
    parser.add_argument('--warmup_steps', type=int, default=10, help='The number of warmup steps to use for fine-tuning.')
    parser.add_argument('--max_steps', type=int, default=700, help='The maximum number of steps to use for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate to use for fine-tuning.')
    parser.add_argument('--fp16', type=str, default='True', help='Whether to use fp16.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='The directory to save the fine-tuned model.')
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit', help='The optimizer to use for fine-tuning.')

    # Evaluation arguments
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='The evaluation strategy to use for fine-tuning.')
    parser.add_argument('--eval_steps', type=int, default=10, help='The number of steps between evaluations.')
    parser.add_argument('--eval_on_test', type=str, default='True', help='Whether to evaluate the model on the test set after fine-tuning.')
    parser.add_argument('--compute_summarization_metrics', type=str, default='True', help='Whether to evaluate the model on ROUGE, BLEU, and BERTScore after fine-tuning.')
    parser.add_argument('--compute_qanda_metrics', type=str, default='False', help='Whether to evaluate the model on QA metrics like F1 and Exact Match (from SQUAD).')
    parser.add_argument('--compute_em_metrics', type=str, default='False', help='Whether to evaluate the model on Accuracy, Precision, Recall, and F1.')

    # Hub arguments
    parser.add_argument('--hub_upload', type=str, default='True', help='Whether to upload the model to the hub.')
    parser.add_argument('--hub_save_id', type=str, default='isaacOnline/opt-125m-peft-summarization', help='The name under which the mode will be saved on the hub.')
    parser.add_argument('--save_steps', type=int, default=10, help='The number of steps between saving the model to the hub.')

    # Generation arguments
    parser.add_argument('--min_new_tokens', type=int, help='The minimum number of new tokens to generate', default=1)
    parser.add_argument('--max_new_tokens', type=int, help='The maximum number of new tokens to generate', default=10)

    # Parse arguments
    args = parser.parse_args()

    # change saving directory
    args.save_dir = 'finetuned_model_'+args.use_model_prompt_defaults
    args.peft_save_dir = 'peft_model_'+args.use_model_prompt_defaults
    args.log_dir = 'logs_'+args.use_model_prompt_defaults
    args.output_dir = 'outputs_'+args.use_model_prompt_defaults
    args.run_name = 'peft_model_'+args.use_model_prompt_defaults

    # Update the start and end prompts if using the model defaults
    if args.use_model_prompt_defaults:
        args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]


    # HF Login
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Create directories if they do not exist
    if not path.exists(args.output_dir):
        makedirs(args.peft_save_dir, exist_ok=True)
        print(f'Created directory {args.peft_save_dir}')


    if not path.exists(args.log_dir):
        mkdir(args.log_dir)
        print(f'Created directory {args.log_dir}')

    # Create a logger
    logger = logging.getLogger(__name__)

    # Setup logging
    print('Setting up logging...')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Use the default log level matching the training args
    log_level = args.log_level.upper()
    logger.setLevel(log_level)

    # Set the log level for the transformers and datasets libraries
    transformers.utils.logging.get_logger("transformers").setLevel(log_level)
    datasets.utils.logging.get_logger("datasets").setLevel(log_level)

    # Log to file
    file_handler = logging.FileHandler(path.join(args.log_dir, f'{args.run_name}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    # Get model and tokenizer
    print('Getting model and tokenizer...')

    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               quantization_type=args.quantization_type,
                                               gradient_checkpointing=bool(args.gradient_checkpointing),
                                               device=args.device)

    logger.info(f'Loaded Model ID: {args.model_id}')

    # Define a data formatter function that wraps the format_data_as_instructions function with the specified arguments
    # ==================
    # chat format
    # ==================
    def data_formatter(data: Mapping,
                       system_message: str=system_message,
                       transaction: str=transaction) -> list[str]:
        """
        Wraps the format_data_as_instructions function with the specified arguments.
        """

        return format_data_as_instructions(data, tokenizer, system_message[args.dataset], transaction[args.dataset])


    # Get LoRA model
    if args.lora == 'True':

        print('Getting LoRA model...')

        if args.tune_modules == 'linear':
            lora_modules = [torch.nn.Linear]
        elif args.tune_modules == 'linear4bit':
            lora_modules = [bnb.nn.Linear4bit]
        elif args.tune_modules == 'linear8bit':
            lora_modules = [bnb.nn.Linear8bit]
        else:
            raise ValueError(f'Invalid tune_modules argument: {args.tune_modules}, must be linear, linear4bit, or linear8bit')

        model = get_lora_model(model,
                               include_modules=lora_modules,
                               exclude_names=args.exclude_names,
                               matrix_rank=32)

        logger.info(f'Loaded LoRA Model')
    
    # Download and prepare data
    print('Downloading and preparing data...')

    data = get_dataset_slices(args.dataset,
                              args.version,
                              train_slice=args.train_slice,
                              validation_slice=args.validation_slice,
                              test_slice=args.test_slice)

    # Get dataset splits
    train_data = data['train']
    
    # Split into training and temp (for validation and testing)
    train_data, temp_data = train_test_split(data['train'], test_size=0.2, random_state=42)
    # Further split temp into validation and test sets
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Set the format of the data
    train_data.set_format(type='torch', device=args.device)
    validation_data.set_format(type='torch', device=args.device)
    test_data.set_format(type='torch', device=args.device)

    logger.info(f'Loaded Dataset: {args.dataset}')

    # Handle no max steps by training one dataset epoch
    if args.max_steps is None:
        args.max_steps = len(train_data)

    # Instantiate trainer
    print('Instantiating trainer...')

    training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            fp16=args.fp16 == 'True',
            output_dir=args.output_dir,
            optim=args.optim,
            use_mps_device=args.use_mps_device == 'True',
            log_level=args.log_level,
            evaluation_strategy=args.evaluation_strategy,
            resume_from_checkpoint=args.resume_from_checkpoint == 'True',
            push_to_hub=args.hub_upload == 'True',
            report_to=['wandb'] if args.wandb_logging == 'True' else [],
            max_steps=args.max_steps,
            logging_first_step=args.logging_first_step == 'True',
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            load_best_model_at_end=True,
        )

    trainer = get_default_trainer(model, 
                                  tokenizer, 
                                  train_data,
                                  eval_dataset=validation_data,
                                  formatting_func=data_formatter,
                                  max_seq_length=args.max_seq_length,
                                  training_args=training_args)
    
    model.config.use_cache = False

    logger.info(f'Instantiated Trainer')

    # Log finetune start time
    if args.wandb_logging == 'True':
        finetune_start_time = time.time()
        wandb.log({'finetune_time_start': finetune_start_time})

    trainer.train()


    # Log finetune end time
    if args.wandb_logging == 'True':
        finetune_end_time = time.time()
        wandb.log({'finetune_time_end': finetune_end_time})
        wandb.log({'finetune_time': finetune_end_time - finetune_start_time})

    logger.info(f'Completed fine-tuning')

    # Save adapter weights and tokenizer
    if args.save_model == 'True':

        print('Saving model and tokenizer...')

        if not path.exists(args.save_dir):
            mkdir(args.save_dir)
            print(f'Created directory {args.save_dir}')

        trainer.model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        logger.info(f'Saved model and tokenizer to {args.save_dir}')


    # Save model to hub
    if args.hub_upload == 'True':

        print('Saving model to hub...')

        trainer.model.push_to_hub(args.hub_save_id, use_auth_token=True)

        logger.info(f'Saved model to hub')

    if args.compute_em_metrics == 'True':

        model = trainer.model

        model.eval()
        model.to(args.device)
        model.config.use_cache = True

        # log eval start time
        if args.wandb_logging == 'True':
            eval_start_time = time.time()
            wandb.log({'inference_time_start': eval_start_time})

        print('Evaluating model on EM Metrics (F1, Precision, Accuracy, Recall)...')
        em_metrics, model_output = evaluate_hf_model_em(model=model,
                                                        tokenizer=tokenizer,
                                                        data=test_data,
                                                        input_column=args.input_col,
                                                        target_column=args.target_col,
                                                        max_samples=200,
                                                        max_tokens=500,
                                                        min_new_tokens=args.min_new_tokens,
                                                        max_new_tokens=args.max_new_tokens,
                                                        remove_suffix=args.suffix,
                                                        run_name=args.run_name,
                                                        system_message=system_message[args.dataset],
                                                        transaction=transaction[args.dataset],
                                                        examples=examples[args.dataset],
                                                        save_output_dir=args.save_dir,
                                                        nshots=0
                                                        )

        # log eval end time
        if args.wandb_logging == 'True':
            eval_end_time = time.time()
            wandb.log({'inference_time_end': eval_end_time})
            wandb.log({'inference_time': eval_end_time - eval_start_time})
        # len(data['test']))

        logger.info('Completed EM Metrics evaluation')
        if args.wandb_logging == 'True':
            wandb.log(em_metrics)
            wandb.log({'Model Output': wandb.Table(dataframe=model_output)})

        # Print metrics
        print('Finetuned Model EM Metrics:')

        for k, v in em_metrics.items():
            print(f'{k}: {v}')

    if args.wandb_logging == 'True':
        wandb.finish()
