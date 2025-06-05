import json
import re
import os
import networkx as nx
from pathlib import Path

# === Load Legal QA Data ===
DATA_DIR = Path("/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/data")
all_files = ["ipc_qa.json", "crpc_qa.json", "constitution_qa.json"]

sections = []

for file in all_files:
    path = DATA_DIR / file
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            sections.append(entry)

# === Initialize Graph ===
G = nx.MultiDiGraph()

# === Utility: Extract section references ===
def extract_cited_sections(text):
    # Simple regex for "Section 123", "section 302"
    return re.findall(r'[Ss]ection\s+(\d+[A-Z]*)', text)

# === Step 1: Add nodes ===
# === Step 1: Add nodes ===
for section in sections:
    # Try to extract section ID from question string
    question = section["question"]
    answer = section["answer"]

    match = re.search(r'section\s+(\d+[A-Z]*)', question, re.IGNORECASE)
    if not match:
        continue

    sec_id = match.group(1)
    node_id = f"SEC_{sec_id}"

    G.add_node(node_id, type="section", question=question, answer=answer)

    # Extract citations
    cited = extract_cited_sections(answer)
    for ref in cited:
        G.add_edge(node_id, f"SEC_{ref}", label="cites")

    # Link keywords
    keywords = re.findall(r'\b[A-Za-z]{5,}\b', question)
    for kw in keywords:
        G.add_node(kw.lower(), type="keyword")
        G.add_edge(kw.lower(), node_id, label="mentions")



# === Save graph ===
import networkx as nx
import pickle

with open("legal_knowledge_graph.gpickle", "wb") as f:
    pickle.dump(G, f)

print(f"✅ Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
