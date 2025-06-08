import json
import re
import pickle
from pathlib import Path

# === Paths ===
DATA_DIR = Path("../data")
GRAPH_PATH = Path("../legal_knowledge_graph.gpickle")
OUTPUT_PATH = DATA_DIR / "legal_qa_instruction.json"
SOURCE_FILES = ["ipc_qa.json", "crpc_qa.json", "constitution_qa.json"]

# === Load Graph ===
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

# === Load QA and Section Texts ===
section_answers = {}
all_entries = []

for file in SOURCE_FILES:
    with open(DATA_DIR / file, "r", encoding="utf-8") as f:
        for item in json.load(f):
            sec_id = item.get("section_id", None)
            if sec_id:
                node_id = f"SEC_{sec_id}"
                section_answers[node_id] = item["answer"]
            all_entries.append(item)

# === Extract context from graph ===
def get_context_for_question(question, top_k=3):
    keywords = re.findall(r'\b[a-zA-Z]{5,}\b', question.lower())
    matched = {}

    for node in G.nodes:
        if G.nodes[node].get("type") == "keyword" and node in keywords:
            for _, sec, data in G.out_edges(node, data=True):
                if data["label"] == "mentions":
                    matched[sec] = matched.get(sec, 0) + 1

    top_sections = sorted(matched.items(), key=lambda x: -x[1])[:top_k]
    context = ""
    for sec, _ in top_sections:
        context += f"- {sec}: {section_answers.get(sec, '')}\n"
    return context.strip()

# === Build instruction data ===
out = []

for entry in all_entries:
    q = entry["question"]
    a = entry["answer"]
    context = get_context_for_question(q)

    if not context.strip():
        continue

    out.append({
        "instruction": "You are a legal assistant. Use the legal context to answer the user's question.",
        "context": context,
        "question": q,
        "answer": a
    })

# === Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"✅ Saved {len(out)} enriched instruction-style entries to {OUTPUT_PATH}")
