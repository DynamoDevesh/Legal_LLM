# graph_prompt_inference.py
import pickle
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

from graph_utils import (
    load_graph_with_embeddings,          # Load G from pickle
    run_rl_agent_traversal,              # RL/DQN or MBRL node selector
    build_prompt_from_path               # Extract relevant IPCs and generate prompt
)

# === 1. Accept your query ===
user_query = input("📝 Enter your legal question: ")

# === 2. Load graph and RL agent ===
G, node_embeddings = load_graph_with_embeddings("legal_kg_curated_large.gpickle", "node_embeddings.pkl")
path_nodes = run_rl_agent_traversal(G, node_embeddings, user_query)

# === 3. Build dynamic prompt ===
from graph_context_generator import build_augmented_prompt

# Extract node info for selected path
context_sections = [G.nodes[n] for n in path_nodes]  # ✅ Fix: extract only the node data
prompt = build_augmented_prompt(user_query, context_sections, top_k=5)




print("\n📌 Final Prompt:\n", prompt)

# === 4. Load LLM and generate answer ===
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ✅ Use full absolute path to your checkpoint folder
model_path = "/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/legal-flan-model/checkpoint_1000"


# ✅ Load from local only — no huggingface hub validation
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False  # Sometimes necessary for older tokenizer.json formats
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    local_files_only=True
)





input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
output = model.generate(input_ids, max_length=256)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n🧠 Legal Answer:\n", answer)
