# pipeline.py

import os
import sys
import argparse
from main import run_main  # Make sure main.py exposes run_main()
from graph_builder import load_nodes_and_edges, build_graph
from graph_encoder import encode_graph_with_gnn
from sentence_transformers import SentenceTransformer
from query_handler import (
    #embed_question,
    find_top_k_nodes,
    expand_neighbors,
    get_node_contexts,
    answer_with_llm,
    log_matched_nodes_and_neighbors, truncate_contexts, rerank_nodes,
    embed_question_with_codebert
)
import logging
from dotenv import load_dotenv
from together import Together

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

Together.api_key = os.getenv("TOGETHER_API_KEY")
print(os.getenv("TOGETHER_API_KEY"))
client = Together.together()

def answer_with_together_llm(question, contexts):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Using the following context:\n\n{contexts}\n\nAnswer the question: {question}"}
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )

    return response.choices[0].message.content.strip()

# CLI
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--url", help="GitHub or GitLab repository URL")
group.add_argument("--local-path", help="Path to a local codebase folder")
args = parser.parse_args()

# Ask user for a question
print("💬 Ask your questions (type 'exit' to quit)\n")


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
embedding_dict = encode_graph_with_gnn(G, dim=768)

# Embed the question
#model = SentenceTransformer("all-MiniLM-L6-v2")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

while True:
    question = input("❓> ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue
    # Step 1: Embed question
    #q_vec = embed_question(question, model)
    q_vec = embed_question_with_codebert(question)

    # Step 2: Retrieve and rerank top nodes
    top_nodes = find_top_k_nodes(q_vec, embedding_dict, k=20)
    node_lookup = {n.id: n for n in nodes}
    top_nodes = rerank_nodes(top_nodes, node_lookup)

    # Step 3: Limit to top N (e.g., 3) before expanding neighbors
    top_ids = [nid for nid, _ in top_nodes[:3]]
    expanded_ids = expand_neighbors(G, top_ids, hops=1)

    # Step 4: Log matched and expanded nodes
    log_matched_nodes_and_neighbors(G, nodes, top_nodes[:5], expanded_ids)

    # Step 5: Fetch and truncate context
    all_contexts = get_node_contexts(expanded_ids, nodes)
    contexts = truncate_contexts(all_contexts, max_tokens=12000)

    # Step 6: LLM answer
    answer = answer_with_together_llm(question, contexts)

    print("\n📌 Answer:")
    print(answer)
    print("\n" + "-" * 50 + "\n")
