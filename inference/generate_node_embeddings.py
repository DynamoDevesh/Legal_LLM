# inference/generate_node_embeddings.py

import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel

print("📦 Loading Knowledge Graph...")
with open("legal_kg_curated_large.gpickle", "rb") as f:
    G = pickle.load(f)

print("🤖 Loading InLegalBERT model (safetensors, trust_remote_code)...")
model_name = "law-ai/InLegalBERT"

# Load tokenizer and transformer model safely
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
transformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Wrap with SentenceTransformer-compatible format
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("🔍 Generating node embeddings...")
node_embeddings = {}
for i, (node_id, attrs) in enumerate(G.nodes(data=True)):
    title = attrs.get("title", "").strip()
    desc = attrs.get("description", "").strip()
    if not title and not desc:
        continue
    text = f"{title}. {desc}".strip() if desc else title
    embedding = model.encode(text)
    node_embeddings[node_id] = embedding

    if i % 100 == 0:
        print(f"🧠 Embedded: {i} nodes")

with open("node_embeddings.pkl", "wb") as f:
    pickle.dump(node_embeddings, f)

print("✅ Done! Saved to node_embeddings.pkl")
