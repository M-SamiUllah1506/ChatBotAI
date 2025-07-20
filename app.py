import gradio as gr
import torch
from transformers import pipeline

# Load the model pipeline
# We don't specify the device; it will auto-detect CPU on the server.
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
)

# This function processes the user's message and chat history
def get_response(message, history):
    # Format the prompt for the model, including previous turns
    messages = [
        {"role": "system", "content": "You are a friendly chatbot."},
    ]
    for turn in history:
        messages.append({"role": "user", "content": turn[0]})
        messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": message})
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate the text
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    # Extract just the new response text to display
    full_response = outputs[0]["generated_text"]
    new_response = full_response[len(prompt):]
    return new_response

# Launch the Gradio Chat Interface
demo = gr.ChatInterface(fn=get_response, title="My Own AI")
demo.launch()
