from parser_python_ts import extract_nodes_and_edges_from_python

def test_embedding(file_path: str):
    print(f"\n🧪 Parsing and embedding: {file_path}\n")
    nodes, edges = extract_nodes_and_edges_from_python(file_path)

    for node in nodes:
        print(f"🔹 Node ID: {node.id}")
        print(f"   Type   : {node.type}")
        print(f"   Name   : {getattr(node, 'name', 'N/A')}")
        print(f"   Embedding (first 5 dims): {node.embedding[:5] if node.embedding else '❌ No embedding'}")
        print("")

if __name__ == "__main__":
    # Change the path below to a small Python file you'd like to test
    test_embedding("example.py")
