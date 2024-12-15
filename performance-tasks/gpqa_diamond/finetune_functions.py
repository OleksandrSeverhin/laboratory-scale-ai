import torch
import bitsandbytes as bnb

import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer
from tqdm import tqdm
from typing import Mapping, Iterable


QUANZATION_MAP = {
    '4bit': BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    '8bit': BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
        torch_dtype=torch.bfloat16,
    ),
}

DEFAULT_TRAINING_ARGS = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        use_mps_device=False,
        log_level='info',
        logging_first_step=True,
        evaluation_strategy='steps',
        eval_steps=25
    )

def format_data_as_instructions(data: Mapping, 
                                tokenizer: AutoTokenizer, 
                                system_message: str='###', 
                                transaction: str='###') -> list[str]:
    """
    Formats text data as instructions for the model. Can be used as a formatting function for the trainer class.
    """

    output_texts = []

    # Iterate over the data and format the text
    for i in tqdm(range(len(data['dialogue'])), desc='Formatting data'):
        
        test_question = f"""\n\n## Content:\n{data['dialogue'][i]}\n\n## Topic:\n{data['section_header'][i]}\n\n## Summary:"""
        test_response = f"""{data['section_text'][i]}"""        
        chat = [
          {"role": "user", "content": system_message + transaction + test_question},
          {"role": "assistant", "content": test_response},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)

    return output_texts

def get_model_and_tokenizer(model_id: str, 
                            quantization_type: str='', 
                            gradient_checkpointing: bool=True, 
                            device: str='auto') -> tuple[AutoModel, AutoTokenizer]:
    """
    Returns a Transformers model and tokenizer for fine-tuning. If quantization_type is provided, the model will be quantized and prepared for training.
    """

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     quantization_config=QUANZATION_MAP[quantization_type], 
                                                     device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     device_map=device)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare the model for training if quantization is requested
    if quantization_type:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def find_lora_modules(model: AutoModel, 
                      include_modules: Iterable=(bnb.nn.Linear4bit), 
                      exclude_names: Iterable=('lm_head')) -> list[str]:
    """
    Returns a list of the modules to be tuned using LoRA.
    """

    # Create a set to store the names of the modules to be tuned
    lora_module_names = set()

    # Iterate over the model and find the modules to be tuned
    for name, module in model.named_modules():

        # Check if the module is in the list of modules to be tuned
        if any(isinstance(module, include_module) for include_module in include_modules):

            # Split the name of the module and add it to the set
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Return the list of module names to be tuned, excluding any names in the exclude list
    return [name for name in list(lora_module_names) if name not in exclude_names]

def get_lora_model(model: AutoModel,
                   matrix_rank: int=8,
                   scaling_factor: int=32,
                   dropout: float=0.05,
                   bias: str='none',
                   task_type: str='CAUSAL_LM',
                   include_modules: Iterable=(bnb.nn.Linear4bit),
                   exclude_names: Iterable=('lm_head')) -> AutoModel:
    """
    Returns a model with LoRA applied to the specified modules.
    """

    config = LoraConfig(
        r=matrix_rank,
        lora_alpha=scaling_factor,
        target_modules=find_lora_modules(model, include_modules, exclude_names),
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )

    return get_peft_model(model, config)


def get_lora_pretrained_model(model_id,
                              adapter_model,
                              device,
                              quantization_type=None,
                              gradient_checkpointing=True,
                              revision=None
                              ):
    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config=QUANZATION_MAP[quantization_type],
                                                     device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     device_map=device)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = PeftModel.from_pretrained(model, adapter_model, revision=revision)

    return model, tokenizer

def get_summarization_dataset(dataset: str,
                              streaming: bool=False,
                              split: str='', 
                              instruction_format: bool=False,
                              input_field: str='article',
                              target_field: str='highlights',
                              start_prompt: str=' ### Summarize the following: ',
                              end_prompt: str=' ### Begin summary: ',
                              suffix: str='',
                              pretokenize: bool=False, 
                              tokenizer: AutoTokenizer=None,
                              max_tokens: int=974) -> dict:
    """
    Returns a dataset for summarization fine-tuning, formatted and tokenized as specified.
    """

    # Download the dataset
    data = load_dataset(dataset, streaming=streaming, split=split)

    # Format the data as instructions if requested
    if instruction_format:
        data = format_data_as_instructions(data, input_field, target_field, start_prompt, end_prompt, suffix)

    # Pretokenize the data if requested
    if pretokenize:
        data = data.map(lambda x: tokenizer(x, truncation=True, max_length=max_tokens), batched=True)

    # Return the dataset
    return data

def get_dataset_slices(dataset: str,
                       version: str='',
                       train_slice: str='train[:1000]',
                       validation_slice: str='validation[:25]',
                       test_slice: str='test[:25]') -> dict:
    """
    Returns a dictionary of subsets of the training, validation, and test splits of a dataset.
    """

    # Download the dataset splits, including the dataset version if specified
    if version:
        train_data = load_dataset(dataset, version=version, split=train_slice)
        validation_data = load_dataset(dataset, version=version, split=validation_slice)
        test_data = load_dataset(dataset, version=version, split=test_slice)
    else:
        train_data = load_dataset(dataset, split=train_slice)
        validation_data = load_dataset(dataset, split=validation_slice)
        test_data = load_dataset(dataset, split=test_slice)

    # Return the dictionary of dataset splits
    return {'train': train_data, 'validation': validation_data, 'test': test_data}
    
def get_dataset_slices_from_xlsx(
    file_path: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> dict:
    """
    Splits a dataset from an Excel file into training, validation, and test sets.

    Args:
        file_path (str): Path to the Excel file containing the dataset.
        train_ratio (float): Proportion of data to include in the training set.
        validation_ratio (float): Proportion of data to include in the validation set.
        test_ratio (float): Proportion of data to include in the test set.

    Returns:
        dict: A dictionary containing the training, validation, and test datasets.
    """
    # Check if ratios sum to 1
    if not (train_ratio + validation_ratio + test_ratio == 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    # Load data from Excel file
    data = pd.read_excel(file_path, engine="openpyxl")

    # Split data into training and temp sets
    train_data, temp_data = train_test_split(
        data, test_size=(1 - train_ratio), random_state=42
    )

    # Calculate validation and test proportions relative to temp set
    validation_ratio_relative = validation_ratio / (validation_ratio + test_ratio)

    # Split temp set into validation and test sets
    validation_data, test_data = train_test_split(
        temp_data, test_size=(1 - validation_ratio_relative), random_state=42
    )

    # Return the dictionary of dataset splits
    return {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }

def get_dataset_dict_slices(dataset_path_or_name, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Load and prepare dataset slices for training, validation, and testing.

    Parameters:
    ----------
    dataset_path_or_name: str or DatasetDict
        Path to the dataset file or dataset identifier.
    train_frac: float, optional
        Fraction of the dataset to use for training. Default is 0.8.
    val_frac: float, optional
        Fraction of the dataset to use for validation. The remainder is used for testing. Default is 0.1.
    seed: int, optional
        Random seed for reproducibility. Default is 42.

    Returns:
    --------
    train_data: Dataset
        The training dataset slice.
    val_data: Dataset
        The validation dataset slice.
    test_data: Dataset
        The testing dataset slice.
    """
    if isinstance(dataset_path_or_name, DatasetDict):
        # If dataset is already loaded as a DatasetDict, use the "train" split
        full_data = dataset_path_or_name["train"]
    else:
        # Assume it is a path or dataset identifier
        dataset = load_dataset(dataset_path_or_name)
        full_data = dataset["train"]

    # Calculate the number of examples for each split
    total_size = len(full_data)
    train_size = int(total_size * train_frac)
    val_size = int(total_size * val_frac)
    test_size = total_size - train_size - val_size

    # Shuffle and split the dataset
    full_data = full_data.shuffle(seed=seed)
    train_data = full_data.select(range(train_size))
    val_data = full_data.select(range(train_size, train_size + val_size))
    test_data = full_data.select(range(train_size + val_size, total_size))

    return train_data, val_data, test_data

def get_default_trainer(model: AutoModel,
                tokenizer: AutoTokenizer,
                train_dataset: Mapping,
                eval_dataset: Mapping=None,
                formatting_func: callable=format_data_as_instructions,                
                max_seq_length: int=974,
                training_args: TrainingArguments=None) -> SFTTrainer:
    """
    Returns the default trainer for fine-tuning a summarization model based on the specified training config.
    """

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args if training_args else DEFAULT_TRAINING_ARGS,
        formatting_func=formatting_func,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=max_seq_length,
        packing=False,
    )

    return trainer
