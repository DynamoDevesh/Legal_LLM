
# evaluate_agent.py
import pickle
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from agent_dqn import DQNAgent
from sklearn.preprocessing import MinMaxScaler

# Load graph and embeddings
with open("legal_kg.gpickle", "rb") as f:
    G = pickle.load(f)

with open("node_embeddings.pkl", "rb") as f:
    node_embeddings = pickle.load(f)

# Load trained agent
state_dim = 768
action_dim = max(len(list(G.successors(n))) for n in G.nodes)

agent = DQNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load("dqn_legal_model.pt"))
agent.q_net.eval()

# Evaluate from a given node
def evaluate_from_node(start_node, steps=10):
    path = [start_node]
    current_node = start_node
    scores = [1.0]  # Start node gets default importance score
    for _ in range(steps):
        neighbors = list(G.successors(current_node))
        if not neighbors:
            break
        state = node_embeddings[current_node].numpy()
        action_idx = agent.select_action(state, list(range(len(neighbors))))
        current_node = neighbors[action_idx]
        path.append(current_node)
        scores.append(float(G.out_degree(current_node)))
    return path, scores

# Visualize path with score-based color intensity
def visualize_path(path, graph, scores):
    subgraph = graph.subgraph(path)
    pos = nx.spring_layout(subgraph, seed=42)
    plt.figure(figsize=(10, 6))
    scaler = MinMaxScaler()
    scores_scaled = scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
    cmap = plt.cm.viridis
    node_colors = [cmap(s) for s in scores_scaled]

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray')
    nx.draw_networkx_labels(subgraph, pos, font_size=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
    sm.set_array([])
    plt.colorbar(sm, label="Node Importance (Degree-based)")
    plt.title("RL Agent Traversed Path with Importance")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
start = "IPC_302"
if start not in G:
    start = list(G.nodes)[0]
path, scores = evaluate_from_node(start)
print("Traversed Path:")
for node in path:
    print(node, "-", G.nodes[node].get("title", "No Title"))

visualize_path(path, G, scores)
