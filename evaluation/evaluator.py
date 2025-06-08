import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from graph.graph_context_generator import build_augmented_prompt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Config ===
CHECKPOINT_DIR = Path("/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/legal-flan-model/checkpoint_1000")
QA_FILE = Path("/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/data/ipc_qa.json")
OUTPUT_FILE = Path("evaluation/evaluated_results.json")
NUM_SAMPLES = 10  # Number of Qs to evaluate

# === Load model and tokenizer ===
print("🔄 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_DIR, local_files_only=True)

# === Load questions ===
with open(QA_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)[:NUM_SAMPLES]

# === Store feedback ===
results = []

for idx, item in enumerate(questions, start=1):
    question = item["question"]
    prompt = build_augmented_prompt(question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n--- Example {idx}/{NUM_SAMPLES} ---")
    print(f"🧾 Question:\n{question}")
    print(f"📚 Contextual Prompt:\n{prompt}")
    print(f"🤖 LLM Answer:\n{answer}")

    try:
        rating = int(input("Rate the answer (1-5): "))
        comment = input("Any comment? (optional): ").strip()
    except KeyboardInterrupt:
        print("\n⛔️ Evaluation stopped.")
        break

    results.append({
        "question": question,
        "prompt": prompt,
        "generated_answer": answer,
        "rating": rating,
        "comment": comment
    })

# === Save results ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(results)} evaluations to {OUTPUT_FILE}")
