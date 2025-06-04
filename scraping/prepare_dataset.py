import json
import os

base_path = "data"
files = ["ipc_qa.json", "crpc_qa.json", "constitution_qa.json"]

formatted_data = []

for fname in files:
    path = os.path.join(base_path, fname)
    print(f"Loading {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for item in entries:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and answer:
            formatted_data.append({
                "text": f"Answer the legal question: {question}\n\nAnswer: {answer}"
            })

# Save the final dataset
output_path = os.path.join(base_path, "legal_qa_sft.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2)

print(f"✅ Saved {len(formatted_data)} entries to {output_path}")
