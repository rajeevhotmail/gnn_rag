# check_tree_sitter.py

import tree_sitter
import sys
import os

print("✅ tree_sitter module loaded from:")
print(tree_sitter.__file__)

print("\n🧩 sys.version:")
print(sys.version)

print("\n🗂️  Contents of module:")
print(dir(tree_sitter))

try:
    from tree_sitter import Language
    print("\n🔍 Language.build_library exists?:", hasattr(Language, 'build_library'))
except Exception as e:
    print("❌ Failed to import Language:", e)
