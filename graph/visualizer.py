import pickle
import matplotlib.pyplot as plt
import networkx as nx
import random

# Load graph
with open("legal_kg.gpickle", "rb") as f:
    G = pickle.load(f)

# -- Option A: Visualize all IPC sections only --
subgraph_nodes = [n for n, attr in G.nodes(data=True) if attr["act"] == "IPC"]

# -- Option B: Random sample of 150 nodes --
# subgraph_nodes = random.sample(list(G.nodes), 150)

# Build subgraph
H = G.subgraph(subgraph_nodes).copy()

# Color by act
color_map = {
    "IPC": "skyblue",
    "CRPC": "orange",
    "CPC": "lightgreen",
    "HMA": "pink",
    "IDA": "plum",
    "IEA": "lightcoral",
    "NIA": "khaki",
    "MVA": "lightgray"
}
node_colors = [color_map.get(G.nodes[n]["act"], "gray") for n in H.nodes]

# Plot
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(H, seed=42, k=0.25)

nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=50, alpha=0.8)
nx.draw_networkx_edges(H, pos, alpha=0.3)

# Optional: labels for a few nodes
sampled = random.sample(list(H.nodes), min(25, len(H.nodes)))
labels = {n: n for n in sampled}
nx.draw_networkx_labels(H, pos, labels, font_size=8)

plt.title("Legal Knowledge Subgraph (e.g., IPC Sections)")
plt.axis("off")
plt.tight_layout()
plt.show()
