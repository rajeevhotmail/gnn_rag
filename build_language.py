# build_language.py

from tree_sitter import Language
print(hasattr(Language, 'build_library'))  # ✅ Should return True

Language.build_library(
  'build/my-languages.so',
  ['vendor/tree-sitter-python']
)
