from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Set the model to evaluation mode
    model.eval()

    print("Welcome to the Llama 3 Chatbot!")
    print("Type 'exit' to end the chat.\n")

    # Initialize the chat history
    chat_history = ""

    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            print("Ending the chat. Goodbye!")
            break

        # Format the input as per Llama-2-chat format
        formatted_input = f"<s>[INST] {user_input.strip()} [/INST]"

        # Append to chat history
        if chat_history:
            chat_history += formatted_input
        else:
            chat_history = formatted_input

        # Tokenize the input
        inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)

        # Generate the model's response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's reply
        # Since we're appending to chat_history, we need to get the new generated part
        assistant_reply = generated_text[len(chat_history):].strip()

        # Print the assistant's reply
        print(f"Llama 3: {assistant_reply}\n")

        # Append the assistant's reply to the chat history for context in the next turn
        chat_history += assistant_reply

if __name__ == "__main__":
    main()