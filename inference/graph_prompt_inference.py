# graph_prompt_inference.py
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration

from graph_utils import (
    load_graph_with_embeddings,          # Load G from pickle
    run_rl_agent_traversal,              # RL/DQN or MBRL node selector
    build_prompt_from_path               # Extract relevant IPCs and generate prompt
)

# === 1. Accept your query ===
user_query = input("📝 Enter your legal question: ")

# === 2. Load graph and RL agent ===
G, node_embeddings = load_graph_with_embeddings("legal_kg.gpickle", "node_embeddings.pkl")
path_nodes = run_rl_agent_traversal(G, node_embeddings, user_query)

# === 3. Build dynamic prompt ===
prompt = build_prompt_from_path(G, path_nodes, user_query)

print("\n📌 Final Prompt:\n", prompt)

# === 4. Load LLM and generate answer ===
model_path = "./legal-flan-model/checkpoint_1000"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
output = model.generate(input_ids, max_length=256)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n🧠 Legal Answer:\n", answer)
