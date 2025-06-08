# inference/graph_utils.py
import torch
import numpy as np

import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer
from agent_dqn import DQNAgent

# === 1. Load graph + embeddings ===
def load_graph_with_embeddings(graph_path, embeddings_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    with open(embeddings_path, "rb") as f:
        node_embeddings = pickle.load(f)
    return G, node_embeddings

# === 2. Embed query using BERT ===
def embed_query(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(query)

# === 3. Run RL-based traversal to get path nodes ===
def run_rl_agent_traversal(G, node_embeddings, user_query):
    state_dim = 768
    action_dim = max(len(list(G.successors(n))) for n in G.nodes)
    
    agent = DQNAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load("dqn_legal_model.pt"))
    agent.q_net.eval()

    query_emb = embed_query(user_query)
    query_node = min(G.nodes, key=lambda n: np.linalg.norm(query_emb - node_embeddings[n]))
    
    # Traverse using RL agent (you can customize how many steps)
    path = [query_node]
    current = query_node
    for _ in range(5):
        state = node_embeddings[current]
        next_idx = agent.select_action(state)
        successors = list(G.successors(current))
        if not successors or next_idx >= len(successors):
            break
        current = successors[next_idx]
        path.append(current)
    return path

# === 4. Build prompt using selected path ===
def build_prompt_from_path(G, path_nodes, user_query):
    prompt = f"Query: {user_query}\nRelevant Legal Sections:\n"
    for node in path_nodes:
        desc = G.nodes[node].get("description", "").strip()
        title = G.nodes[node].get("title", "").strip()
        if desc:
            prompt += f"{node}: {desc}\n"
        elif title:
            prompt += f"{node}: {title}\n"
    prompt += "Answer:"
    return prompt
