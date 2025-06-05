import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from graph.graph_context_generator import build_augmented_prompt

# Load model and tokenizer from trained path
model_path = "/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/legal-flan-model/checkpoint_1000"  # adjust if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Build prompt using graph
    prompt = build_augmented_prompt(user_input)
    print("\n[DEBUG] Prompt fed to LLM:\n", prompt)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("🤖 Bot:", answer)
