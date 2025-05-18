# query_handler.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import networkx as nx
import tiktoken
def truncate_contexts(contexts: list, max_tokens: int = 12000) -> list:
    enc = tiktoken.encoding_for_model("gpt-4")
    total = 0
    limited = []

    for chunk in contexts:
        tokens = enc.encode(chunk)
        if total + len(tokens) > max_tokens:
            break
        limited.append(chunk)
        total += len(tokens)

    return limited
def log_matched_nodes_and_neighbors(graph, nodes, top_nodes, expanded_ids):
    id_to_node = {n.id: n for n in nodes}

    print("\n🔍 Top matched nodes:")
    for node_id, score in top_nodes:
        node = id_to_node.get(node_id)
        if node:
            name = node.name or "<unnamed>"
            typ = node.metadata.get("type", "") if node.metadata else "unknown"
            print(f"  🟢 {name} ({typ}) - score: {score:.3f}")
        else:
            print(f"  ⚠️ Node {node_id} not found in loaded nodes (skipped)")

    print("\n🔗 Neighbors included via 1-hop expansion:")
    for nid in expanded_ids:
        if nid not in [n[0] for n in top_nodes]:
            node = id_to_node.get(nid)
            if node:
                name = node.name or "<unnamed>"
                typ = node.metadata.get("type", "") if node.metadata else "unknown"
                print(f"  🔸 {name} ({typ})")


def expand_neighbors(graph: nx.DiGraph, node_ids: list, hops: int = 1) -> list:
    expanded = set(node_ids)
    for node_id in node_ids:
        neighbors = nx.single_source_shortest_path_length(graph, node_id, cutoff=hops)
        expanded.update(neighbors.keys())
    return list(expanded)

def embed_question(question: str, model) -> np.ndarray:
    return model.encode(question, convert_to_numpy=True)

def find_top_k_nodes(question_vec, node_embeddings: dict, k=5):
    node_ids = list(node_embeddings.keys())
    node_vecs = np.array([node_embeddings[n] for n in node_ids])

    similarities = cosine_similarity([question_vec], node_vecs)[0]
    top_indices = similarities.argsort()[-k:][::-1]

    return [(node_ids[i], similarities[i]) for i in top_indices]

def get_node_contexts(node_ids: list, all_nodes: list):
    return [n.content for n in all_nodes if n.id in node_ids]

def answer_with_llm(question: str, contexts: list, llm_provider="openai", api_key=None):
    context_text = "\n\n".join(contexts)
    prompt = f"""You are a programming assistant.

Given the following code context:

{context_text}

Answer the following question:

Q: {question}
A:"""

    if llm_provider == "openai":
        import openai
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    else:
        raise NotImplementedError("Only OpenAI is supported for now")
