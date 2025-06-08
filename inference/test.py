import pickle
G = pickle.load(open("legal_kg.gpickle", "rb"))

# Check if sections have useful descriptions
for node in G.nodes:
    if G.nodes[node].get("type") == "section":
        print("✅", node)
        print("Title:", G.nodes[node].get("title"))
        print("Description:", G.nodes[node].get("description"))
        break
