# parser_python_ts.py

from tree_sitter import Language, Parser
from utils.types import Node, Edge
import os
from typing import List
from codebert_embedder import CodeBERTEmbedder

embedder = CodeBERTEmbedder()

# Load the Python grammar (update path if needed)
LANGUAGE = Language('build/my-languages.so', 'python')
parser = Parser()
parser.set_language(LANGUAGE)


def extract_nodes_and_edges_from_python(file_path: str) -> (List[Node], List[Edge]):
    """
    Parses a Python file with Tree-sitter and extracts nodes and edges.

    Returns:
        nodes: List of Node
        edges: List of Edge
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    nodes = []
    edges = []

    def traverse(node, parent_id=None):
        if node.type == "function_definition":
            func_name = get_node_text(node.child_by_field_name("name"), code)
            func_id = f"{file_path}::function::{func_name}"
            func_content = get_code_slice(code, node)
            embedding = embedder.embed(func_content).squeeze(0).tolist() if func_content.strip() else [0.0] * 768
            nodes.append(Node(
                id=func_id,
                type="function",
                name=func_name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=func_content,
                embedding=embedding,
            ))

            if parent_id:
                edges.append(Edge(source=parent_id, target=func_id, relation="defines"))

        elif node.type == "class_definition":
            class_name = get_node_text(node.child_by_field_name("name"), code)
            class_id = f"{file_path}::class::{class_name}"
            class_content = get_code_slice(code, node)
            embedding = embedder.embed(class_content).squeeze(0).tolist() if class_content.strip() else [0.0] * 768
            nodes.append(Node(
                id=class_id,
                type="class",
                name=class_name,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=class_content,
                embedding=embedding,
            ))

            if parent_id:
                edges.append(Edge(source=parent_id, target=class_id, relation="defines"))

            # Add inheritance edges
            super_node = node.child_by_field_name("superclasses")
            if super_node:
                for child in super_node.children:
                    if child.type == "identifier":
                        super_name = get_node_text(child, code)
                        edges.append(Edge(source=class_id, target=super_name, relation="inherits"))

        # Traverse children
        for child in node.children:
            traverse(child, parent_id=parent_id)

    # Add file as a node
    file_id = f"{file_path}::file"
    embedding = embedder.embed(code).squeeze(0).tolist() if code.strip() else [0.0] * 768
    nodes.append(Node(
        id=file_id,
        type="file",
        name=os.path.basename(file_path),
        file_path=file_path,
        content=code,
        embedding=embedding,
    ))

    traverse(root, parent_id=file_id)
    return nodes, edges


def get_node_text(node, code: str) -> str:
    return code[node.start_byte:node.end_byte]


def get_code_slice(code: str, node) -> str:
    lines = code.splitlines()
    return "\n".join(lines[node.start_point[0]: node.end_point[0] + 1])
