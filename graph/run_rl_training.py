# run_rl_training.py
import pickle
import random
import torch
import networkx as nx
import numpy as np
from agent_dqn import DQNAgent
from tqdm import trange

# Load graph and embeddings
with open("legal_kg.gpickle", "rb") as f:
    G = pickle.load(f)

with open("node_embeddings.pkl", "rb") as f:
    node_embeddings = pickle.load(f)

# Filter nodes with outgoing edges only
valid_nodes = [n for n in G.nodes if list(G.successors(n)) and n in node_embeddings]

# Environment settings
MAX_STEPS = 10
EPISODES = 300

# Agent setup
state_dim = 768
action_dim = max(len(list(G.successors(n))) for n in valid_nodes)
agent = DQNAgent(state_dim, action_dim)

def get_state(node):
    return node_embeddings[node].numpy()

def select_action_idx(agent, state, neighbors):
    indices = list(range(len(neighbors)))
    return agent.select_action(state, indices)

# Define dummy reward logic (e.g., reward high degree nodes)
def compute_reward(node):
    if G.out_degree(node) > 5:
        return 1.0
    else:
        return -0.1

print("[INFO] Starting training...")

for ep in trange(EPISODES):
    current_node = random.choice(valid_nodes)
    total_reward = 0

    for step in range(MAX_STEPS):
        state = get_state(current_node)
        neighbors = list(G.successors(current_node))
        if not neighbors:
            break

        action_idx = select_action_idx(agent, state, neighbors)
        next_node = neighbors[action_idx]
        reward = compute_reward(next_node)
        next_state = get_state(next_node)
        done = (step == MAX_STEPS - 1)

        agent.store_transition(state, action_idx, reward, next_state, done)
        agent.optimize_model()

        current_node = next_node
        total_reward += reward

    if ep % 10 == 0:
        agent.update_target_network()
        print(f"Episode {ep} total reward: {total_reward:.2f}")

# Save model
torch.save(agent.q_net.state_dict(), "dqn_legal_model.pt")
print("[DONE] Training complete.")
