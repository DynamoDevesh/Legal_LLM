import networkx as nx
import re
import json
from pathlib import Path
from difflib import get_close_matches

# === Load Graph and Legal Section Texts ===
GRAPH_PATH = Path("/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/legal_knowledge_graph.gpickle")
DATA_DIR = Path("/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/data")
TEXT_SOURCES = ["ipc_qa.json", "crpc_qa.json", "constitution_qa.json"]

import pickle
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)


# === Load Section Answers for Context Retrieval ===
section_texts = {}
for file in TEXT_SOURCES:
    with open(DATA_DIR / file, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            sec_id = entry.get("section_id", None)
            if sec_id:
                node_id = f"SEC_{sec_id}"
                section_texts[node_id] = entry["answer"]

# === Keyword Extraction ===
def extract_keywords(text):
    return re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())

# === Get related section nodes using mentions or similarity ===
def find_related_sections(question, top_k=3):
    keywords = extract_keywords(question)
    matched_sections = {}

    for node in G.nodes:
        if G.nodes[node].get("type") == "keyword" and node in keywords:
            # Follow edges to sections
            for _, sec, data in G.out_edges(node, data=True):
                if data["label"] == "mentions":
                    matched_sections[sec] = matched_sections.get(sec, 0) + 1

    # Fallback: string similarity on section_texts
    if not matched_sections:
        for kw in keywords:
            for node in section_texts.keys():
                score = sum([kw in node.lower(), kw in section_texts[node].lower()])
                if score > 0:
                    matched_sections[node] = matched_sections.get(node, 0) + score

    # Return top k
    return sorted(matched_sections.items(), key=lambda x: -x[1])[:top_k]

# === Build augmented prompt ===
def build_augmented_prompt(question, top_k=3):
    context_sections = find_related_sections(question, top_k=top_k)
    context = ""
    for sec_id, _ in context_sections:
        law_text = section_texts.get(sec_id)
        if not law_text:
            node_data = G.nodes.get(sec_id, {})
            law_text = node_data.get("answer") or node_data.get("question") or "❓ No text available"
        context += f"- {sec_id}: {law_text.strip()[:400]}...\n"

    prompt = f"""You are a legal advisor.

Use the following legal context to answer the question:

{context}
Question: {question}
Answer:"""
    return prompt


# === Example Usage ===
if __name__ == "__main__":
    sample_q = "Can someone be arrested without a warrant?"
    final_prompt = build_augmented_prompt(sample_q)
    print(final_prompt)
