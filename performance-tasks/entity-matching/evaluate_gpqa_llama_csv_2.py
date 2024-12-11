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

from datasets import Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from os import path, mkdir, getenv, makedirs
from typing import Mapping
from sklearn.model_selection import train_test_split

from finetune_functions import get_model_and_tokenizer, get_lora_model, get_default_trainer, evaluate_hf_model_em


def format_data_as_instructions(data: Mapping, tokenizer: AutoTokenizer, nshots=0) -> list[str]:
    """Formats text data as instructions for the model."""
    output_texts = []

    for idx in range(len(data['prompt'])):
        question = data['prompt'][idx]
        chat = [{"role": "user", "content": question}]
        chat.append({"role": "assistant", "content": data['overall_label'][idx]})
        output_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        output_texts.append(output_text)

    return output_texts


def evaluate_answers(model, tokenizer, data, input_col, target_col):
    correct = 0
    total = len(data)
    
    for idx in range(total):
        input_text = data[input_col][idx]
        expected_answer = data[target_col][idx]
        
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_answer.strip() == expected_answer.strip():
            correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy


if __name__ == '__main__':
    hf_login()
    data_path = '/home/ubuntu/laboratory-scale-ai/data/gpqa_diamond_15.xlsx'

    parser = argparse.ArgumentParser(description='Evaluate LLaMA-3.2b with questions from an XLSX file.')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2b', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')

    args = parser.parse_args()

    device = args.device

    # HF Login
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_id, device=device)

    # Load questions and answers from Excel
    df = pd.read_excel(data_path)

    # Convert DataFrame to Dataset
    data = Dataset.from_pandas(df)

    # Evaluate model
    accuracy = evaluate_answers(model, tokenizer, data, input_col='question', target_col='answer')

    # Print summary
    print(f'Accuracy: {accuracy:.2f}%')