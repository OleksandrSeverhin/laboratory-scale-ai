import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer

def evaluate_answers(model, tokenizer, data, input_col, target_col, batch_size=4):
    correct = 0
    total = len(data)
    
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_data = data[input_col][start_idx:end_idx]
        batch_targets = data[target_col][start_idx:end_idx]

        inputs = tokenizer(list(batch_data), return_tensors="pt", padding=True, truncation=True).to(device)

        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for gen_ans, expected_ans in zip(generated_answers, batch_targets):
            if gen_ans.strip() == expected_ans.strip():
                correct += 1
    
    accuracy = (correct / total) * 100
    return accuracy

if __name__ == '__main__':
    hf_login()
    data_paths = [
        '/home/ubuntu/laboratory-scale-ai/data/gpqa_diamond_15.xlsx',
        '/home/ubuntu/laboratory-scale-ai/data/gpqa_diamond_15_o1_mini.xlsx',
        '/home/ubuntu/laboratory-scale-ai/data/gpqa_diamond_15_o1_preview.xlsx'
    ]

    parser = argparse.ArgumentParser(description='Evaluate LLaMA-3.1b with questions from an XLSX file.')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='The model ID to fine-tune.')
    parser.add_argument('--hf_token_var', type=str, default='HF_TOKEN', help='Name of the HuggingFace API token variable name.')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to mount the model on.')

    args = parser.parse_args()
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        offload_folder="./offload",
        offload_state_dict=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for data_path in data_paths:
        df = pd.read_excel(data_path)
        df['Correct Answer'] = df['Correct Answer'].astype(str)
        data = Dataset.from_pandas(df)
        accuracy = evaluate_answers(model, tokenizer, data, input_col='Question', target_col='Correct Answer', batch_size=8)
        print(f'{data_path}: Accuracy: {accuracy:.2f}%')
