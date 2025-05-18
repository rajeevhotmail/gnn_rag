# codebert_embedder.py

from transformers import RobertaTokenizer, RobertaModel
import torch

class CodeBERTEmbedder:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.model.eval()

    def embed(self, code: str, max_tokens: int = 256) -> torch.Tensor:
        tokens = self.tokenizer(code, return_tensors="pt", truncation=True, padding="max_length", max_length=max_tokens)
        with torch.no_grad():
            output = self.model(**tokens)
        return output.last_hidden_state[:, 0, :]  # [CLS] token embedding
