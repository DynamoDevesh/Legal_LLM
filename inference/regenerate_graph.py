import json
import pickle
import networkx as nx

GRAPH_PATH = "legal_kg_curated_large.gpickle"
DATA_PATH = "/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/graph/curated_legal_dataset_large.json"

print("📥 Loading curated synthetic dataset...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("🔧 Building legal knowledge graph...")
G = nx.DiGraph()

for entry in data:
    node_id = entry["node_id"]  # ← correct key based on your format
    G.add_node(
        node_id,
        title=entry.get("title", ""),
        description=entry.get("description", ""),
        act=entry.get("act", "UNKNOWN"),
        type=entry.get("type", "section")
    )

print(f"✅ Total nodes added: {len(G.nodes)}")

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)

print(f"📦 Graph saved to {GRAPH_PATH}")
