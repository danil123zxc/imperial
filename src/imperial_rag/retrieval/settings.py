from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from imperial_rag.config import env_float, env_int, env_str


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int = 400
    chunk_overlap: int = 50
    vector_fetch_k: int = 70
    vector_k: int = 70
    keyword_limit: int = 30
    rerank_input_limit: int = 100
    rerank_top_n: int = 10
    mmr_lambda_mult: float = 0.4
    rrf_k: int = 60
    primary_reranker: str = "dashscope:qwen3-rerank"
    fallback_reranker: str = "fallback:deterministic"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        qwen_rerank_model = env_str("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen3-rerank")
        primary_reranker = env_str("IMPERIAL_RAG_PRIMARY_RERANKER", f"dashscope:{qwen_rerank_model}")
        return cls(
            chunk_size=env_int("IMPERIAL_RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=env_int("IMPERIAL_RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            vector_fetch_k=env_int("IMPERIAL_RAG_VECTOR_FETCH_K", cls.vector_fetch_k),
            vector_k=env_int("IMPERIAL_RAG_VECTOR_K", cls.vector_k),
            keyword_limit=env_int("IMPERIAL_RAG_KEYWORD_LIMIT", cls.keyword_limit),
            rerank_input_limit=env_int("IMPERIAL_RAG_RERANK_INPUT_LIMIT", cls.rerank_input_limit),
            rerank_top_n=env_int("IMPERIAL_RAG_RERANK_TOP_N", cls.rerank_top_n),
            mmr_lambda_mult=env_float("IMPERIAL_RAG_MMR_LAMBDA_MULT", cls.mmr_lambda_mult),
            rrf_k=env_int("IMPERIAL_RAG_RRF_K", cls.rrf_k),
            primary_reranker=primary_reranker,
            fallback_reranker=env_str("IMPERIAL_RAG_FALLBACK_RERANKER", cls.fallback_reranker),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
