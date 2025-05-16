# file_walker.py

import os
from typing import List

def find_files_by_extension(repo_path: str, extensions: List[str]) -> List[str]:
    """
    Recursively find files in repo_path that match the given extensions.

    Args:
        repo_path: Root path of the repo
        extensions: List of extensions (e.g., ['.py', '.java'])

    Returns:
        List of file paths
    """
    matched_files = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matched_files.append(os.path.join(root, file))

    return matched_files
