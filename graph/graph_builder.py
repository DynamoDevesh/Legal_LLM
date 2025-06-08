import os
import json
import networkx as nx
import pickle

GRAPH_DIR = "/Users/dynamodevesh/Desktop/Legal_Bot/Legal_LLM/Indian-Law-Penal-Code-Json-main"  # current directory
DATA_FILES = [
    ("ipc.json", "IPC"),
    ("crpc.json", "CRPC"),
    ("cpc.json", "CPC"),
    ("hma.json", "HMA"),
    ("ida.json", "IDA"),
    ("iea.json", "IEA"),
    ("nia.json", "NIA"),
    ("MVA.json", "MVA")
]

def parse_hma_entry(raw_str):
    try:
        # Skip blank lines or headers
        if not raw_str.strip() or "chapter,section," in raw_str:
            return None
        parts = raw_str.split(",", 3)
        if len(parts) < 4:
            return None
        return {
            "chapter": parts[0].strip(),
            "section": parts[1].strip(),
            "section_title": parts[2].strip(),
            "section_desc": parts[3].strip().strip('"')
        }
    except:
        return None

def parse_section_id(act, section):
    return (
        section.get("section") or
        section.get("Section") or
        section.get("section_no") or
        section.get("id")
    )

def extract_title(section):
    return (
        section.get("section_title") or
        section.get("title") or
        section.get("section_heading") or
        section.get("heading") or
        ""
    )

def extract_desc(section):
    return (
        section.get("section_desc") or
        section.get("description") or
        section.get("text") or
        ""
    )

def build_graph():
    G = nx.DiGraph()

    for filename, act in DATA_FILES:
        filepath = os.path.join(GRAPH_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename} not found.")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")
                continue

        node_count = 0

        if act == "HMA":
            # HMA is a list of dicts with single CSV string key
            for row in data:
                csv_str = row.get("chapter,section,section_title,section_desc", "")
                parsed = parse_hma_entry(csv_str)
                if not parsed:
                    continue
                raw_id = parsed.get("section")
                if not raw_id:
                    continue
                section_id = f"{act}_{str(raw_id).strip()}"
                G.add_node(section_id,
                           act=act,
                           title=parsed.get("section_title", "").strip(),
                           description=parsed.get("section_desc", "").strip())
                node_count += 1
        else:
            for section in data:
                raw_id = parse_section_id(act, section)
                if not raw_id:
                    continue
                section_id = f"{act}_{str(raw_id).strip()}"
                G.add_node(section_id,
                           act=act,
                           title=extract_title(section).strip(),
                           description=extract_desc(section).strip())
                node_count += 1

        print(f"[INFO] Added {node_count} sections from {act}")

    return G

if __name__ == "__main__":
    G = build_graph()
    with open("legal_kg.gpickle", "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Saved graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
