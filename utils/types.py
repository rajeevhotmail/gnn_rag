# utils/types.py

from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Node:
    id: str
    type: str  # class, function, file, etc.
    name: str
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class Edge:
    source: str
    target: str
    relation: str  # e.g., calls, defines, inherits, imports
