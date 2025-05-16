# query_handler.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

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
