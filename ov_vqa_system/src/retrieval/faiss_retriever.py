from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


RetrievedPassage = Dict[str, Any]


@dataclass
class RetrieverConfig:
    """
    FAISS retriever configuration.
    """

    index_path: str
    corpus_path: str
    embedding_model: str = "BAAI/bge-m3"
    top_k: int = 10
    device: str = "cuda"
    normalize_embeddings: bool = True
    query_prefix: str = ""


class FaissRetriever:
    """
    Dense retriever based on FAISS and SentenceTransformer embeddings.
    """

    def __init__(
        self,
        index_path: str | os.PathLike,
        corpus_path: str | os.PathLike,
        embedding_model: str = "BAAI/bge-m3",
        top_k: int = 10,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        lazy_load: bool = False,
    ) -> None:
        """
        Args:
            index_path: Path to FAISS index.
            corpus_path: Path to passage mapping JSON file.
            embedding_model: SentenceTransformer model name or local path.
            top_k: Number of passages to retrieve.
            device: Device for embedding model, e.g. cuda or cpu.
            normalize_embeddings: Whether to L2-normalize query embeddings.
            query_prefix: Optional prefix added before every query.
            lazy_load: If True, delay loading until first retrieval.
        """
        self.config = RetrieverConfig(
            index_path=str(index_path),
            corpus_path=str(corpus_path),
            embedding_model=embedding_model,
            top_k=top_k,
            device=device,
            normalize_embeddings=normalize_embeddings,
            query_prefix=query_prefix,
        )

        self.index = None
        self.passages = None
        self.embedder = None

        if not lazy_load:
            self.load()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "FaissRetriever":
        """
        Build retriever from experiment config.

        Args:
            cfg: Full experiment config.

        Returns:
            FaissRetriever.
        """
        retrieval_cfg = cfg.get("retrieval", {})

        return cls(
            index_path=retrieval_cfg.get("index_path"),
            corpus_path=retrieval_cfg.get("corpus_path"),
            embedding_model=retrieval_cfg.get("embedding_model", "BAAI/bge-m3"),
            top_k=int(retrieval_cfg.get("top_k", 10)),
            device=retrieval_cfg.get("device", "cuda"),
            normalize_embeddings=bool(
                retrieval_cfg.get("normalize_embeddings", True)
            ),
            query_prefix=retrieval_cfg.get("query_prefix", ""),
        )

    def load(self) -> None:
        """
        Load FAISS index, passage mapping, and embedding model.
        """
        self._load_index()
        self._load_passages()
        self._load_embedder()

    def _load_index(self) -> None:
        """
        Load FAISS index.
        """
        index_path = Path(self.config.index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Please install faiss-gpu or faiss-cpu."
            ) from e

        self.index = faiss.read_index(str(index_path))

    def _load_passages(self) -> None:
        """
        Load passage mapping file.
        """
        corpus_path = Path(self.config.corpus_path)

        if not corpus_path.exists():
            raise FileNotFoundError(f"Passage mapping not found: {corpus_path}")

        with corpus_path.open("r", encoding="utf-8") as f:
            passages = json.load(f)

        if not isinstance(passages, (dict, list)):
            raise ValueError(
                "Passage mapping must be either dict or list. "
                f"Got {type(passages)} from {corpus_path}"
            )

        self.passages = passages

    def _load_embedder(self) -> None:
        """
        Load SentenceTransformer embedding model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please install it before running retrieval."
            ) from e

        self.embedder = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device,
        )

    def _ensure_loaded(self) -> None:
        """
        Ensure all resources are loaded.
        """
        if self.index is None or self.passages is None or self.embedder is None:
            self.load()

    def _format_query(self, query: str) -> str:
        """
        Apply optional query prefix.

        Args:
            query: Raw query.

        Returns:
            Formatted query.
        """
        query = str(query).strip()

        if self.config.query_prefix:
            return f"{self.config.query_prefix}{query}"

        return query

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into a dense vector.

        Args:
            query: Query text.

        Returns:
            Query embedding of shape [1, dim], dtype float32.
        """
        self._ensure_loaded()

        formatted_query = self._format_query(query)

        embedding = self.embedder.encode(
            [formatted_query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype="float32")

        if embedding.ndim == 1:
            embedding = embedding[None, :]

        return embedding

    def _get_passage_by_id(self, idx: int) -> RetrievedPassage:
        """
        Get passage by FAISS index id.

        Args:
            idx: FAISS result id.

        Returns:
            Passage dictionary.
        """
        if self.passages is None:
            raise RuntimeError("Passages are not loaded.")

        idx_int = int(idx)

        if isinstance(self.passages, list):
            if idx_int < 0 or idx_int >= len(self.passages):
                return {
                    "id": idx_int,
                    "title": "",
                    "text": "",
                    "missing": True,
                }

            item = self.passages[idx_int]

        else:
            key = str(idx_int)
            item = self.passages.get(key)

            if item is None:
                item = self.passages.get(idx_int)

            if item is None:
                return {
                    "id": idx_int,
                    "title": "",
                    "text": "",
                    "missing": True,
                }

        if isinstance(item, str):
            return {
                "id": idx_int,
                "title": "",
                "text": item,
            }

        if isinstance(item, dict):
            title = (
                item.get("title")
                or item.get("page_title")
                or item.get("wiki_title")
                or ""
            )

            text = (
                item.get("text")
                or item.get("passage")
                or item.get("content")
                or item.get("paragraph")
                or ""
            )

            output = dict(item)
            output["id"] = output.get("id", idx_int)
            output["title"] = title
            output["text"] = text

            return output

        return {
            "id": idx_int,
            "title": "",
            "text": str(item),
        }

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        """
        Retrieve top-k passages for a query.

        Args:
            query: Query text.
            top_k: Optional override for number of retrieved passages.

        Returns:
            List of retrieved passage dictionaries.
        """
        self._ensure_loaded()

        if not query or not str(query).strip():
            return []

        top_k = int(top_k or self.config.top_k)

        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(query_embedding, top_k)

        scores = scores[0].tolist()
        indices = indices[0].tolist()

        results: List[RetrievedPassage] = []

        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if int(idx) < 0:
                continue

            passage = self._get_passage_by_id(int(idx))

            passage["rank"] = rank
            passage["score"] = float(score)
            passage["retrieval_score"] = float(score)
            passage["source"] = "wikipedia"
            passage["retriever"] = "faiss"

            results.append(passage)

        return results

    def retrieve_for_sample(
        self,
        sample: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        question = sample.get("question", "")

        return self.search(
            query=question,
            top_k=top_k,
        )

    def check_index_mapping_consistency(self) -> Dict[str, Any]:
        self._ensure_loaded()

        index_total = int(self.index.ntotal)
        index_dim = int(self.index.d)

        if isinstance(self.passages, list):
            mapping_size = len(self.passages)
            mapping_type = "list"
        else:
            mapping_size = len(self.passages)
            mapping_type = "dict"

        return {
            "index_path": self.config.index_path,
            "corpus_path": self.config.corpus_path,
            "index_total": index_total,
            "index_dim": index_dim,
            "mapping_size": mapping_size,
            "mapping_type": mapping_type,
            "same_size": index_total == mapping_size,
        }

    def __call__(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        return self.search(query=query, top_k=top_k)


class EmptyRetriever:

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        return []

    def retrieve_for_sample(
        self,
        sample: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        return []

    def __call__(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        return []


def build_retriever(cfg: Dict[str, Any]):
    retrieval_cfg = cfg.get("retrieval", {})
    enabled = bool(retrieval_cfg.get("enabled", True))

    if not enabled:
        return EmptyRetriever()

    return FaissRetriever.from_config(cfg)