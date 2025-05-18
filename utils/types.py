# utils/types.py

from dataclasses import dataclass
from typing import  Dict
from typing import List, Optional

@dataclass
class Node:
    id: str
    type: str  # class, function, file, etc.
    file_path: str
    name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content: Optional[str] = None
    metadata: Optional[Dict] = None
    embedding: Optional[List[float]] = None

@dataclass
class Edge:
    source: str
    target: str
    relation: str  # e.g., calls, defines, inherits, imports
