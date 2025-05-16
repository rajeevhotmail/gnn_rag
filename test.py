from graph_builder import load_nodes_and_edges, build_graph
from graph_encoder import encode_graph_with_gnn
from query_handler import embed_question, find_top_k_nodes, get_node_contexts, answer_with_llm
from sentence_transformers import SentenceTransformer

# Load graph
nodes, edges = load_nodes_and_edges(
    "output/psf_requests/data/graph_nodes.json",
    "output/psf_requests/data/graph_edges.json"
)
G = build_graph(nodes, edges)
embedding_dict = encode_graph_with_gnn(G)

# Question
question = input("🔍 Enter your question: ")

# Embed question
question_model = SentenceTransformer("all-MiniLM-L6-v2")
q_vec = embed_question(question, question_model)

# Match top nodes
top_nodes = find_top_k_nodes(q_vec, embedding_dict, k=5)
top_ids = [nid for nid, _ in top_nodes]
contexts = get_node_contexts(top_ids, nodes)

# Answer
answer = answer_with_llm(
    question=question,
    contexts=contexts,
    llm_provider="openai",
    api_key=""
)

print("\n📌 Answer:")
print(answer)
