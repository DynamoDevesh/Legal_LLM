import pickle
import networkx as nx
import re

ACT_ALIASES = {
    "ipc": "IPC",
    "crpc": "CRPC",
    "cpc": "CPC",
    "hma": "HMA",
    "ida": "IDA",
    "iea": "IEA",
    "nia": "NIA",
    "mva": "MVA"
}

# Load existing graph
with open("legal_kg.gpickle", "rb") as f:
    G = pickle.load(f)

ref_regex = re.compile(r"section\s+(\d+[A-Z]?)\s*(?:of\s+(?:the\s+)?([a-zA-Z]+))?", re.IGNORECASE)

added_edges = 0

for source_node, data in G.nodes(data=True):
    src_act = data["act"]
    desc = data.get("description", "")
    if not desc:
        continue

    for match in ref_regex.finditer(desc.lower()):
        sec_num, act_hint = match.groups()
        target_act = ACT_ALIASES.get(act_hint.lower(), src_act) if act_hint else src_act
        target_id = f"{target_act}_{sec_num.upper()}"

        if target_id in G:
            G.add_edge(source_node, target_id, type="refers_to")
            added_edges += 1

print(f"[INFO] Added {added_edges} inferred 'refers_to' edges.")

# Save updated graph
with open("legal_kg.gpickle", "wb") as f:
    pickle.dump(G, f)

print(f"[DONE] Updated graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
