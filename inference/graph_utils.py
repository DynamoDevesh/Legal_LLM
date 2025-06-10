# inference/graph_utils.py
import torch
import numpy as np
import os
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
    from sentence_transformers import SentenceTransformer

    # Replace model
    model = SentenceTransformer("law-ai/InLegalBERT")

    return model.encode(query)

# === 3. Run RL-based traversal to get path nodes ===
def run_rl_agent_traversal(G, node_embeddings, user_query):
    state_dim = 768
    #action_dim = max([len(list(G.neighbors(n))) for n in G.nodes if len(list(G.neighbors(n))) > 0], default=25)
    action_dim =25
    print(f"📏 RL Agent Config → state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgent(state_dim, action_dim)
    model_path = os.path.join(os.path.dirname(__file__), "dqn_legal_model.pt")
    agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.q_net.eval()

    query_emb = embed_query(user_query)
    query_emb = query_emb.cpu().numpy() if hasattr(query_emb, "cpu") else query_emb
    node_embeddings = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in node_embeddings.items()}

    # Keep only nodes that exist in both graph and embeddings
    legal_nodes = list(set(G.nodes).intersection(set(node_embeddings.keys())))
    node_embeddings = {k: v for k, v in node_embeddings.items() if k in G.nodes}

    print(f"🔍 Total nodes in G: {len(G.nodes)}")
    print(f"✅ Legal nodes with embeddings: {len(legal_nodes)}")

    if not legal_nodes:
        raise ValueError("🚨 No nodes found with embeddings. Check your graph or embedding loading.")

    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = query_emb.reshape(1, -1)
    node_keys = list(node_embeddings.keys())
    node_vecs = np.array([node_embeddings[k] for k in node_keys])
    sims = cosine_similarity(query_vec, node_vecs)[0]
    ranked_nodes = [node_keys[i] for i in sims.argsort()[::-1]]

    # Use top node as start point
    query_node = ranked_nodes[0]
    path = [query_node]
    current = query_node

    for _ in range(5):
        state = node_embeddings[current]
        legal_actions = list(G.neighbors(current))

        if not legal_actions:
            break

        next_idx = agent.select_action(state, legal_actions)
        if next_idx >= len(legal_actions):
            break

        current = legal_actions[next_idx]
        path.append(current)

    return path


# === 4. Build prompt using selected path ===
def build_prompt_from_path(G, path_nodes, user_query):
    prompt = f"""You are a legal assistant.
Use the following legal context to answer the user's query accurately.

User Query: {user_query}

Relevant Legal Sections:\n"""
    
    for node in path_nodes:
        desc = G.nodes[node].get("description", "").strip()
        title = G.nodes[node].get("title", "").strip()
        label = f"{node}: {desc or title}"
        prompt += f"- {label}\n"

    prompt += "\nPlease provide a concise legal answer in simple terms.\nAnswer:"
    return prompt
