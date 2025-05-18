# test.py

from graph_builder import load_nodes_and_edges, build_graph
from graph_encoder import encode_graph_with_gnn
from query_handler import (
    embed_question,
    find_top_k_nodes,
    get_node_contexts,
    answer_with_llm,
    expand_neighbors, log_matched_nodes_and_neighbors,
)
from sentence_transformers import SentenceTransformer
import os
import sys
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
    print("💡 Set it using: $env:OPENAI_API_KEY = 'sk-...' (PowerShell)")
    sys.exit(1)
# Config
NODE_FILE = "output\\monty_context_service\\data\\graph_nodes.json"
EDGE_FILE = "output\\monty_context_service\\data\\graph_edges.json"

K = 5  # top-k nodes
HOPS = 1  # graph hops

# Load graph
nodes, edges = load_nodes_and_edges(NODE_FILE, EDGE_FILE)
G = build_graph(nodes, edges)
embedding_dict = encode_graph_with_gnn(G, dim=384)

# Question input
question = input("🔍 Enter your question: ")

# Embed question
model = SentenceTransformer("all-MiniLM-L6-v2")
q_vec = embed_question(question, model)

# Find top-k matching nodes
top_nodes = find_top_k_nodes(q_vec, embedding_dict, k=K)
top_ids = [nid for nid, _ in top_nodes]

# Expand with neighbors
expanded_ids = expand_neighbors(G, top_ids, hops=HOPS)
log_matched_nodes_and_neighbors(G, nodes, top_nodes, expanded_ids)

# Get content from nodes
contexts = get_node_contexts(expanded_ids, nodes)

# Get answer from LLM
answer = answer_with_llm(
    question=question,
    contexts=contexts,
    llm_provider="openai",
    api_key=OPENAI_KEY
)

print("\n📌 Answer:")
print(answer)
