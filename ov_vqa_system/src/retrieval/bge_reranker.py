from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


Passage = Dict[str, Any]


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5
    device: str = "cuda"
    max_length: int = 512
    local_files_only: bool = True

class BGEReranker:
    """
    BGE cross-encoder reranker using transformers backend.

    This avoids FlagEmbedding tokenizer compatibility issues.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
        device: str = "cuda",
        max_length: int = 512,
        local_files_only: bool = True,
        lazy_load: bool = False,
    ) -> None:
        self.config = RerankerConfig(
            model_name=model_name,
            top_k=top_k,
            device=device,
            max_length=max_length,
            local_files_only=local_files_only,
        )

        self.model = None
        self.tokenizer = None

        if not lazy_load:
            self._load_model()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BGEReranker":
        reranker_cfg = cfg.get("reranker", {})

        return cls(
            model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-base"),
            top_k=int(reranker_cfg.get("top_k", 5)),
            device=reranker_cfg.get("device", "cuda"),
            max_length=int(reranker_cfg.get("max_length", 512)),
            local_files_only=bool(reranker_cfg.get("local_files_only", True)),
        )

    def _load_model(self) -> None:
        """
        Load BGE reranker with transformers.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
            use_fast=False,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
        )

        self.model.to(self.config.device)
        self.model.eval()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self._load_model()

    @staticmethod
    def _passage_to_text(passage: Passage) -> str:
        title = passage.get("title", "")
        text = (
            passage.get("text")
            or passage.get("passage")
            or passage.get("content")
            or ""
        )

        title = str(title).strip()
        text = str(text).strip()

        if title and text:
            return f"{title}. {text}"

        if text:
            return text

        return title

    def score(
        self,
        query: str,
        passages: Sequence[Passage],
    ) -> List[float]:
        """
        Score passages with respect to a query.
        """
        import torch

        self._ensure_loaded()

        if not passages:
            return []

        scores = []

        with torch.no_grad():
            for passage in passages:
                passage_text = self._passage_to_text(passage)

                inputs = self.tokenizer(
                    query,
                    passage_text,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                inputs = {
                    key: value.to(self.config.device)
                    for key, value in inputs.items()
                }

                outputs = self.model(**inputs)
                score = outputs.logits.view(-1)[0].detach().float().cpu().item()
                scores.append(float(score))

        return scores

    def rerank(
        self,
        query: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        if not passages:
            return []

        top_k = int(top_k or self.config.top_k)

        scores = self.score(query=query, passages=passages)

        reranked = []

        for passage, score in zip(passages, scores):
            item = dict(passage)
            item["rerank_score"] = float(score)
            item["source"] = item.get("source", "wikipedia")
            item["reranker"] = "bge_transformers"
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank

        return reranked[:top_k]

    def rerank_for_sample(
        self,
        sample: Dict[str, Any],
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        question = sample.get("question", "")

        return self.rerank(
            query=question,
            passages=passages,
            top_k=top_k,
        )

    def __call__(
        self,
        query: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        return self.rerank(
            query=query,
            passages=passages,
            top_k=top_k,
        )

class IdentityReranker:
    """
    Fallback reranker used when reranking is disabled.

    It keeps the original FAISS ranking and truncates to top_k.
    """

    def __init__(self, top_k: int = 5) -> None:
        """
        Args:
            top_k: Number of passages to keep.
        """
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        """
        Return original passages with preserved order.

        Args:
            query: Unused.
            passages: Retrieved passages.
            top_k: Optional top-k override.

        Returns:
            Truncated passage list.
        """
        top_k = int(top_k or self.top_k)

        output = []

        for rank, passage in enumerate(passages[:top_k], start=1):
            item = dict(passage)
            item["rerank_rank"] = rank
            item["rerank_score"] = item.get("retrieval_score", item.get("score", 0.0))
            item["reranker"] = "identity"
            output.append(item)

        return output

    def rerank_for_sample(
        self,
        sample: Dict[str, Any],
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        """
        Rerank passages for a unified VQA sample.

        Args:
            sample: Unified VQA sample.
            passages: Retrieved passages.
            top_k: Optional top-k override.

        Returns:
            Truncated passage list.
        """
        question = sample.get("question", "")

        return self.rerank(
            query=question,
            passages=passages,
            top_k=top_k,
        )

    def __call__(
        self,
        query: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
    ) -> List[Passage]:
        """
        Alias for rerank().
        """
        return self.rerank(
            query=query,
            passages=passages,
            top_k=top_k,
        )


def build_reranker(cfg: Dict[str, Any]):
    """
    Build reranker from config.

    Args:
        cfg: Full experiment config.

    Returns:
        BGEReranker or IdentityReranker.
    """
    reranker_cfg = cfg.get("reranker", {})
    enabled = bool(reranker_cfg.get("enabled", True))
    top_k = int(reranker_cfg.get("top_k", 5))

    if not enabled:
        return IdentityReranker(top_k=top_k)

    return BGEReranker.from_config(cfg)