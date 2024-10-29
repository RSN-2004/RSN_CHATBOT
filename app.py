from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])       
def chat():
    # Handle both GET and POST requests
    msg = request.args.get("msg") if request.method == "GET" else request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    # Get the response from the chatbot
    response = get_Chat_response(msg)
    return jsonify({"response": response})

def get_Chat_response(text):
    # Initialize chat history
    chat_history_ids = None
    
    # Encode the new user input and add the eos_token
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last output tokens from bot and return the response
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
