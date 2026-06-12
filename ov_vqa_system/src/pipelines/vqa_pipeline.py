from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.detector.yolo_wrapper import build_detector
from src.llm.ollama_wrapper import build_llm
from src.prompts.concept_prompts import build_concept_generator
from src.reasoning.candidate_generator import build_candidate_generator
from src.reasoning.final_reasoner import build_final_reasoner
from src.retrieval.bge_reranker import build_reranker
from src.retrieval.faiss_retriever import build_retriever


@dataclass
class PipelineResult:
    """
    Full output of one VQA inference.

    Attributes:
        question_id:
            Dataset question id.
        image_id:
            Dataset image id.
        question:
            Question text.
        pred_answer:
            Human-readable final answer.
        clean_answer:
            Normalized answer for evaluation.
        raw_reasoning_output:
            Raw LLM final reasoning output.
        concepts:
            Concepts generated from question.
        expanded_concepts:
            Concepts after knowledge expansion.
        selected_prompts:
            Detection prompts sent to YOLO-World.
        detections:
            Raw detector outputs.
        retrieved_passages:
            FAISS retrieved passages.
        reranked_passages:
            Reranked passages.
        answer_candidates:
            Generated answer candidates.
        visual_evidence:
            Formatted visual evidence used by final reasoner.
        knowledge_evidence:
            Formatted knowledge evidence used by final reasoner.
        reasoning_input:
            Structured JSON-like evidence input.
        final_prompt:
            Final prompt sent to LLM.
        raw_concept_generation_output:
            Raw LLM output for concept generation / expansion.
        raw_prompt_selection_output:
            Raw LLM output for prompt selection.
        raw_candidate_generation_output:
            Raw LLM output for answer candidate generation.
        candidate_generation_prompt:
            Prompt used for answer candidate generation.
        latency:
            Time cost for each stage.
        error:
            Error message if inference failed.
    """

    question_id: Any
    image_id: Any
    question: str

    pred_answer: str
    clean_answer: str
    raw_reasoning_output: str

    concepts: List[str]
    expanded_concepts: List[str]
    selected_prompts: List[str]
    detections: List[Dict[str, Any]]

    retrieved_passages: List[Dict[str, Any]]
    reranked_passages: List[Dict[str, Any]]

    answer_candidates: List[str]
    visual_evidence: List[Dict[str, Any]]
    knowledge_evidence: List[Dict[str, Any]]

    reasoning_input: str
    final_prompt: str

    raw_concept_generation_output: str
    raw_prompt_selection_output: str
    raw_candidate_generation_output: str
    candidate_generation_prompt: str

    latency: Dict[str, float]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary result.
        """
        return asdict(self)


class OVVQAPipeline:
    """
    Retrieval-augmented open-vocabulary VQA pipeline.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        detector: Optional[Any] = None,
        concept_generator: Optional[Any] = None,
        candidate_generator: Optional[Any] = None,
        final_reasoner: Optional[Any] = None,
    ) -> None:
        """
        Args:
            cfg: Full experiment config.
            llm: Optional prebuilt LLM.
            retriever: Optional prebuilt retriever.
            reranker: Optional prebuilt reranker.
            detector: Optional prebuilt detector.
            concept_generator: Optional prebuilt concept generator.
            candidate_generator: Optional prebuilt candidate generator.
            final_reasoner: Optional prebuilt final reasoner.
        """
        self.cfg = cfg

        self.llm = llm or build_llm(cfg)
        self.retriever = retriever or build_retriever(cfg)
        self.reranker = reranker or build_reranker(cfg)

        detector_enabled = bool(cfg.get("detector", {}).get("enabled", True))
        self.detector = detector or (build_detector(cfg) if detector_enabled else None)

        self.concept_generator = concept_generator or build_concept_generator(
            cfg=cfg,
            llm=self.llm,
        )

        self.candidate_generator = candidate_generator or build_candidate_generator(
            cfg=cfg,
            llm=self.llm,
        )

        self.final_reasoner = final_reasoner or build_final_reasoner(
            cfg=cfg,
            llm=self.llm,
        )

        self.retrieval_enabled = bool(cfg.get("retrieval", {}).get("enabled", True))
        self.reranker_enabled = bool(cfg.get("reranker", {}).get("enabled", True))
        self.detector_enabled = detector_enabled

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "OVVQAPipeline":
        """
        Build pipeline from config.

        Args:
            cfg: Full experiment config.

        Returns:
            OVVQAPipeline.
        """
        return cls(cfg=cfg)

    @staticmethod
    def _elapsed(start_time: float) -> float:
        """
        Compute elapsed seconds.

        Args:
            start_time: Start timestamp.

        Returns:
            Elapsed seconds.
        """
        return round(time.time() - start_time, 6)

    @staticmethod
    def _empty_result(
        sample: Dict[str, Any],
        error: Optional[str] = None,
    ) -> PipelineResult:
        """
        Build an empty result, mainly used when an error occurs.

        Args:
            sample: Unified VQA sample.
            error: Optional error message.

        Returns:
            PipelineResult with empty fields.
        """
        return PipelineResult(
            question_id=sample.get("question_id"),
            image_id=sample.get("image_id"),
            question=sample.get("question", ""),
            pred_answer="",
            clean_answer="",
            raw_reasoning_output="",
            concepts=[],
            expanded_concepts=[],
            selected_prompts=[],
            detections=[],
            retrieved_passages=[],
            reranked_passages=[],
            answer_candidates=[],
            visual_evidence=[],
            knowledge_evidence=[],
            reasoning_input="",
            final_prompt="",
            raw_concept_generation_output="",
            raw_prompt_selection_output="",
            raw_candidate_generation_output="",
            candidate_generation_prompt="",
            latency={},
            error=error,
        )

    def retrieve(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge passages.

        Args:
            sample: Unified VQA sample.

        Returns:
            Retrieved passages.
        """
        if not self.retrieval_enabled:
            return []

        top_k = int(self.cfg.get("retrieval", {}).get("top_k", 10))

        return self.retriever.retrieve_for_sample(
            sample=sample,
            top_k=top_k,
        )

    def rerank(
        self,
        sample: Dict[str, Any],
        retrieved_passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved passages.

        Args:
            sample: Unified VQA sample.
            retrieved_passages: Retrieved passages.

        Returns:
            Reranked passages.
        """
        if not retrieved_passages:
            return []

        top_k = int(self.cfg.get("reranker", {}).get("top_k", 5))

        return self.reranker.rerank_for_sample(
            sample=sample,
            passages=retrieved_passages,
            top_k=top_k,
        )

    def generate_concepts(
        self,
        sample: Dict[str, Any],
        passages: List[Dict[str, Any]],
    ):
        """
        Generate concepts and final detection prompts.

        Args:
            sample: Unified VQA sample.
            passages: Knowledge passages for concept expansion.

        Returns:
            ConceptResult.
        """
        return self.concept_generator.run(
            question=sample.get("question", ""),
            passages=passages,
        )

    def detect(
        self,
        sample: Dict[str, Any],
        selected_prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run open-vocabulary detection.

        Args:
            sample: Unified VQA sample.
            selected_prompts: Detection prompts.

        Returns:
            Detector outputs.
        """
        if not self.detector_enabled or self.detector is None:
            return []

        if not selected_prompts:
            return []

        return self.detector.detect_from_sample(
            sample=sample,
            prompts=selected_prompts,
        )

    def generate_answer_candidates(
        self,
        sample: Dict[str, Any],
        selected_prompts: List[str],
        detections: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate answer candidates.

        Args:
            sample: Unified VQA sample.
            selected_prompts: Detection prompts.
            detections: Detector outputs.
            passages: Knowledge passages.

        Returns:
            Candidate generation result.
        """
        # For A-OKVQA future use: if choices exist, they can be treated as candidates.
        choices = sample.get("choices") or []

        candidate_result = self.candidate_generator.generate_for_sample(
            sample=sample,
            selected_prompts=selected_prompts,
            detections=detections,
            passages=passages,
        )

        candidates = candidate_result.get("candidates", [])

        if choices:
            merged = []
            seen = set()

            for item in list(choices) + list(candidates):
                item = str(item).strip()
                if not item:
                    continue

                key = item.lower()
                if key in seen:
                    continue

                seen.add(key)
                merged.append(item)

            candidate_result["candidates"] = merged

        return candidate_result

    def reason(
        self,
        sample: Dict[str, Any],
        detections: List[Dict[str, Any]],
        passages: List[Dict[str, Any]],
        answer_candidates: List[str],
        selected_prompts: List[str],
    ):
        """
        Run final answer reasoning.

        Args:
            sample: Unified VQA sample.
            detections: Detector outputs.
            passages: Knowledge passages.
            answer_candidates: Answer candidates.
            selected_prompts: Detection prompts.

        Returns:
            FinalReasoningResult.
        """
        return self.final_reasoner.reason_for_sample(
            sample=sample,
            detections=detections,
            passages=passages,
            answer_candidates=answer_candidates,
            selected_prompts=selected_prompts,
        )

    def run(
        self,
        sample: Dict[str, Any],
        catch_errors: bool = False,
    ) -> PipelineResult:
        """
        Run full inference on one sample.

        Args:
            sample: Unified VQA sample.
            catch_errors:
                If True, return an empty result with error message instead of raising.
                For debugging, keep False. For long experiments, True can be useful.

        Returns:
            PipelineResult.
        """
        try:
            latency: Dict[str, float] = {}
            total_start = time.time()

            # 1. Retrieval
            start = time.time()
            retrieved_passages = self.retrieve(sample)
            latency["retrieval"] = self._elapsed(start)

            # 2. Reranking
            start = time.time()
            reranked_passages = self.rerank(sample, retrieved_passages)
            latency["reranking"] = self._elapsed(start)

            # If reranker is disabled, IdentityReranker already returns top-k.
            # If retrieval is enabled but reranking returns empty for any reason,
            # fall back to retrieved passages.
            knowledge_for_downstream = reranked_passages or retrieved_passages

            # 3. Concept generation and prompt selection
            start = time.time()
            concept_result = self.generate_concepts(
                sample=sample,
                passages=knowledge_for_downstream,
            )
            latency["concept_generation"] = self._elapsed(start)

            selected_prompts = concept_result.selected_prompts

            # 4. Detection
            start = time.time()
            detections = self.detect(
                sample=sample,
                selected_prompts=selected_prompts,
            )
            latency["detection"] = self._elapsed(start)

            # 5. Answer candidate generation
            start = time.time()
            candidate_result = self.generate_answer_candidates(
                sample=sample,
                selected_prompts=selected_prompts,
                detections=detections,
                passages=knowledge_for_downstream,
            )
            latency["candidate_generation"] = self._elapsed(start)

            answer_candidates = candidate_result.get("candidates", [])

            # 6. Final evidence-based reasoning
            start = time.time()
            reasoning_result = self.reason(
                sample=sample,
                detections=detections,
                passages=knowledge_for_downstream,
                answer_candidates=answer_candidates,
                selected_prompts=selected_prompts,
            )
            latency["final_reasoning"] = self._elapsed(start)

            latency["total"] = self._elapsed(total_start)

            return PipelineResult(
                question_id=sample.get("question_id"),
                image_id=sample.get("image_id"),
                question=sample.get("question", ""),
                pred_answer=reasoning_result.pred_answer,
                clean_answer=reasoning_result.clean_answer,
                raw_reasoning_output=reasoning_result.raw_reasoning_output,
                concepts=concept_result.raw_concepts,
                expanded_concepts=concept_result.expanded_concepts,
                selected_prompts=selected_prompts,
                detections=detections,
                retrieved_passages=retrieved_passages,
                reranked_passages=reranked_passages,
                answer_candidates=answer_candidates,
                visual_evidence=reasoning_result.visual_evidence,
                knowledge_evidence=reasoning_result.knowledge_evidence,
                reasoning_input=reasoning_result.reasoning_input,
                final_prompt=reasoning_result.final_prompt,
                raw_concept_generation_output=concept_result.raw_generation_output,
                raw_prompt_selection_output=concept_result.raw_selection_output,
                raw_candidate_generation_output=candidate_result.get("raw_output", ""),
                candidate_generation_prompt=candidate_result.get("prompt", ""),
                latency=latency,
                error=None,
            )

        except Exception as e:
            if catch_errors:
                result = self._empty_result(
                    sample=sample,
                    error=str(e),
                )
                return result

            raise

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        catch_errors: bool = False,
    ) -> List[PipelineResult]:
        """
        Run inference on a batch of samples.

        Args:
            samples: List of unified VQA samples.
            catch_errors: Whether to catch per-sample errors.

        Returns:
            List of PipelineResult.
        """
        results = []

        for sample in samples:
            result = self.run(
                sample=sample,
                catch_errors=catch_errors,
            )
            results.append(result)

        return results

    def __call__(
        self,
        sample: Dict[str, Any],
        catch_errors: bool = False,
    ) -> PipelineResult:
        """
        Alias for run().
        """
        return self.run(
            sample=sample,
            catch_errors=catch_errors,
        )


def build_vqa_pipeline(cfg: Dict[str, Any]) -> OVVQAPipeline:
    """
    Build OV-VQA pipeline from config.

    Args:
        cfg: Full experiment config.

    Returns:
        OVVQAPipeline.
    """
    return OVVQAPipeline.from_config(cfg)