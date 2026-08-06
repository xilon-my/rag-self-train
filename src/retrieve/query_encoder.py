"""Unified query-encoder wrapper — THE invariant of the whole project.

The bge s2p query instruction is applied to QUERY text only; passages never get
it. Frozen and fine-tuned rows, training and inference, all go through this exact
wrapper, so the off-the-shelf vs self-trained comparison differs only in the
weight tensors — never in prompt conventions.
"""
from sentence_transformers import SentenceTransformer

DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class QueryEncoder:
    def __init__(self, model_name: str, device: str = "cpu",
                 query_instruction: str = DEFAULT_QUERY_INSTRUCTION, batch_size: int = 32):
        self.model = SentenceTransformer(model_name, device=device)
        self.instruction = query_instruction
        self.batch_size = batch_size
        self.device = device

    def encode(self, texts: list[str], is_query: bool, normalize: bool = True) -> "numpy.ndarray":
        if is_query:
            texts = [self.instruction + t for t in texts]
        return self.model.encode(
            texts, normalize_embeddings=normalize, batch_size=self.batch_size, show_progress_bar=False
        )
