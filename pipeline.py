# pipeline.py

import os
import sys
import argparse
from main import run_main  # Make sure main.py exposes run_main()
from graph_builder import load_nodes_and_edges, build_graph
from graph_encoder import encode_graph_with_gnn
from query_handler import (
    embed_question,
    find_top_k_nodes,
    expand_neighbors,
    get_node_contexts,
    answer_with_llm,
    log_matched_nodes_and_neighbors, truncate_contexts,
)
from sentence_transformers import SentenceTransformer

# CLI
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--url", help="GitHub or GitLab repository URL")
group.add_argument("--local-path", help="Path to a local codebase folder")
args = parser.parse_args()

# Ask user for a question
question = input("🔍 Enter your question: ").strip()
if not question:
    print("❌ No question entered.")
    sys.exit(1)

# Determine repo name
if args.url:
    repo_name = args.url.strip("/").split("/")[-2] + "_" + args.url.strip("/").split("/")[-1]
    run_main(url=args.url, local_path=None, persistent=True, role="programmer", skip_process=False)
elif args.local_path:
    repo_name = os.path.basename(os.path.normpath(args.local_path))
    run_main(url=None, local_path=args.local_path, persistent=True, role="programmer", skip_process=False)

# Paths
base_path = f"output/{repo_name}/data"
node_path = os.path.join(base_path, "graph_nodes.json")
edge_path = os.path.join(base_path, "graph_edges.json")

# Load graph
nodes, edges = load_nodes_and_edges(node_path, edge_path)
G = build_graph(nodes, edges)

# Encode nodes
embedding_dict = encode_graph_with_gnn(G, dim=384)

# Embed the question
model = SentenceTransformer("all-MiniLM-L6-v2")
q_vec = embed_question(question, model)

# Retrieve + expand
top_nodes = find_top_k_nodes(q_vec, embedding_dict, k=5)
top_ids = [nid for nid, _ in top_nodes]
expanded_ids = expand_neighbors(G, top_ids, hops=1)

# Log matched nodes
log_matched_nodes_and_neighbors(G, nodes, top_nodes, expanded_ids)

# Fetch context
all_contexts = get_node_contexts(expanded_ids, nodes)
contexts = truncate_contexts(all_contexts, max_tokens=12000)

# Answer with OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

answer = answer_with_llm(
    question=question,
    contexts=contexts,
    llm_provider="openai",
    api_key=api_key
)

print("\n📌 Answer:")
print(answer)
