"""
Concept generation and prompt selection for OV-VQA.

This module implements the concept side of the pipeline:

1. Generate latent visual concepts from the question.
2. Optionally expand concepts using retrieved knowledge.
3. Select a compact set of visually detectable prompts.
4. Merge heuristic fallback prompts to avoid empty / biased detections.

Supported strategies:
- question
- noun_phrases
- llm_concepts
- llm_concepts_knowledge
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.utils.text import clean_concepts, split_llm_list_output


@dataclass
class ConceptResult:
    """
    Concept generation result.
    """

    raw_concepts: List[str]
    expanded_concepts: List[str]
    selected_prompts: List[str]
    raw_generation_output: str = ""
    raw_selection_output: str = ""


def format_knowledge_passages(
    passages: Optional[Sequence[Dict[str, Any]]],
    max_passages: int = 5,
    max_chars_per_passage: int = 500,
) -> str:
    """
    Format retrieved passages for concept expansion.
    """
    if not passages:
        return "None"

    lines = []

    for idx, passage in enumerate(passages[:max_passages], start=1):
        if isinstance(passage, dict):
            text = (
                passage.get("text")
                or passage.get("passage")
                or passage.get("content")
                or ""
            )
            title = passage.get("title")
        else:
            text = str(passage)
            title = None

        text = str(text).strip()

        if not text:
            continue

        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage].rstrip() + "..."

        if title:
            lines.append(f"{idx}. {title}: {text}")
        else:
            lines.append(f"{idx}. {text}")

    return "\n".join(lines) if lines else "None"


def build_concept_generation_prompt(
    question: str,
    max_concepts: int = 12,
) -> str:
    """
    Build prompt for generating latent visual concepts from question.
    """
    return f"""
Given a visual question, infer short concrete visual concepts that may be useful for open-vocabulary object detection.

The concepts should be:
- visible or potentially visible in an image
- short noun phrases
- concrete objects, attributes, scene elements, materials, tools, animals, places, or actions
- useful for grounding the question visually

Avoid:
- long explanations
- abstract answers
- full sentences
- concepts that cannot be visually detected
- meta text such as "Here are the concepts"

Question:
{question}

Return at most {max_concepts} concepts.
Return only one concept per line.
Do not include explanations.
""".strip()


def build_knowledge_expansion_prompt(
    question: str,
    raw_concepts: Sequence[str],
    knowledge_text: str,
    max_concepts: int = 12,
) -> str:
    """
    Build prompt for expanding concepts using retrieved knowledge.
    """
    concepts_text = "\n".join(f"- {c}" for c in raw_concepts) or "None"

    return f"""
You are given a visual question, initial visual concepts, and retrieved knowledge.

Your task is to expand the concept list into concrete visual prompts for open-vocabulary detection.

Keep concepts that are visually detectable.
Add knowledge-derived concepts only if they are likely to appear in the image or help visually ground the question.

Question:
{question}

Initial concepts:
{concepts_text}

Retrieved knowledge:
{knowledge_text}

Return at most {max_concepts} concrete visual concepts.
Return only one concept per line.
Do not include explanations.
Do not include meta text such as "Here are".
""".strip()


def build_prompt_selection_prompt(
    question: str,
    candidate_concepts: Sequence[str],
    max_selected: int = 5,
) -> str:
    """
    Build prompt for selecting final detection prompts.
    """
    candidates_text = "\n".join(
        f"{idx}. {concept}"
        for idx, concept in enumerate(candidate_concepts, start=1)
    )

    return f"""
Select the best visual detection prompts for answering the question.

A good detection prompt should be:
- visually detectable
- concrete
- relevant to the question
- short
- not redundant with another selected prompt

Question:
{question}

Candidate concepts:
{candidates_text}

Select at most {max_selected} prompts.
Return only one selected prompt per line.
Do not include explanations.
Do not include meta text such as "Here are the prompts".
""".strip()


def question_as_prompt(question: str, max_selected: int = 5) -> List[str]:
    """
    Use the whole question as detection prompt.
    """
    question = str(question).strip()
    return [question] if question else []


def extract_noun_phrases_heuristic(
    question: str,
    max_phrases: int = 5,
) -> List[str]:
    """
    Lightweight noun-like phrase extraction without external NLP dependencies.
    """
    if question is None:
        return []

    text = question.lower().strip()
    text = re.sub(r"[?!.;,]", " ", text)
    text = re.sub(r"\s+", " ", text)

    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "in",
        "on",
        "at",
        "of",
        "for",
        "to",
        "with",
        "from",
        "by",
        "about",
        "the",
        "a",
        "an",
        "and",
        "or",
        "be",
        "being",
        "been",
        "it",
        "they",
        "he",
        "she",
        "we",
        "you",
        "i",
        "likely",
        "probably",
        "usually",
        "name",
        "type",
    }

    words = text.split()

    phrases = []
    buffer = []

    for word in words:
        if word in stop_words:
            if buffer:
                phrases.append(" ".join(buffer))
                buffer = []
        else:
            buffer.append(word)

    if buffer:
        phrases.append(" ".join(buffer))

    cleaned = []

    for phrase in phrases:
        phrase = phrase.strip()
        phrase = re.sub(r"\s+", " ", phrase)

        if len(phrase) <= 1:
            continue

        if phrase in stop_words:
            continue

        cleaned.append(phrase)

    return clean_concepts(cleaned, max_items=max_phrases)


def heuristic_fallback_prompts(question: str) -> List[str]:
    """
    Add question-type specific fallback prompts.

    These prompts help when the question itself does not mention the visible object,
    e.g. "What sport can you use this for?"
    """
    if question is None:
        return []

    q = question.lower()
    prompts: List[str] = []

    if "sport" in q or "game" in q or "playing" in q:
        prompts.extend(
            [
                "person",
                "ball",
                "bat",
                "racket",
                "motorcycle",
                "bike",
                "helmet",
                "court",
                "field",
                "track",
                "net",
                "sports equipment",
            ]
        )

    if "plant" in q or "grow" in q or "flower" in q:
        prompts.extend(
            [
                "plant",
                "flower",
                "tree",
                "leaf",
                "vine",
                "stem",
                "root",
                "ground",
                "soil",
                "pot",
            ]
        )

    if "animal" in q:
        prompts.extend(
            [
                "animal",
                "dog",
                "cat",
                "horse",
                "cow",
                "bird",
                "mouth",
                "leg",
                "tail",
            ]
        )

    if "kitchen" in q:
        prompts.extend(
            [
                "kitchen",
                "counter",
                "island",
                "table",
                "sink",
                "cabinet",
                "stove",
            ]
        )

    if "toilet" in q or "bathroom" in q:
        prompts.extend(
            [
                "toilet",
                "bathroom",
                "man",
                "person",
                "floor",
            ]
        )

    if "bag" in q or "carrying" in q:
        prompts.extend(
            [
                "person",
                "bag",
                "red bag",
                "clothes",
                "food",
                "lunch",
                "shoes",
            ]
        )

    if "bat" in q:
        prompts.extend(
            [
                "person",
                "bat",
                "baseball bat",
                "ball",
                "swing",
                "baseball",
            ]
        )

    if "toy" in q:
        prompts.extend(
            [
                "toy",
                "stuffed animal",
                "teddy bear",
                "doll",
                "robot",
                "bear",
            ]
        )

    if "place" in q or "go to this place" in q:
        prompts.extend(
            [
                "store",
                "shop",
                "museum",
                "building",
                "street",
                "sign",
                "business",
            ]
        )

    # General fallback for underspecified questions.
    prompts.extend(
        [
            "person",
            "object",
            "vehicle",
            "animal",
            "food",
            "plant",
            "tool",
            "furniture",
        ]
    )

    return clean_concepts(prompts, max_items=30)


class ConceptPromptBuilder:
    """
    Builds concept-generation and prompt-selection prompts.
    """

    def __init__(
        self,
        max_raw_concepts: int = 12,
        max_selected_concepts: int = 5,
        max_knowledge_passages: int = 5,
    ) -> None:
        self.max_raw_concepts = max_raw_concepts
        self.max_selected_concepts = max_selected_concepts
        self.max_knowledge_passages = max_knowledge_passages

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ConceptPromptBuilder":
        concept_cfg = cfg.get("concept", {})

        return cls(
            max_raw_concepts=int(concept_cfg.get("max_raw_concepts", 12)),
            max_selected_concepts=int(concept_cfg.get("max_selected_concepts", 5)),
            max_knowledge_passages=int(concept_cfg.get("max_knowledge_passages", 5)),
        )

    def concept_generation_prompt(self, question: str) -> str:
        return build_concept_generation_prompt(
            question=question,
            max_concepts=self.max_raw_concepts,
        )

    def knowledge_expansion_prompt(
        self,
        question: str,
        raw_concepts: Sequence[str],
        passages: Optional[Sequence[Dict[str, Any]]],
    ) -> str:
        knowledge_text = format_knowledge_passages(
            passages=passages,
            max_passages=self.max_knowledge_passages,
        )

        return build_knowledge_expansion_prompt(
            question=question,
            raw_concepts=raw_concepts,
            knowledge_text=knowledge_text,
            max_concepts=self.max_raw_concepts,
        )

    def prompt_selection_prompt(
        self,
        question: str,
        candidate_concepts: Sequence[str],
    ) -> str:
        return build_prompt_selection_prompt(
            question=question,
            candidate_concepts=candidate_concepts,
            max_selected=self.max_selected_concepts,
        )


class ConceptGenerator:
    """
    End-to-end concept generator.

    Strategies:
        question
        noun_phrases
        llm_concepts
        llm_concepts_knowledge
    """

    def __init__(
        self,
        llm: Any,
        strategy: str = "llm_concepts_knowledge",
        enabled: bool = True,
        max_raw_concepts: int = 12,
        max_selected_concepts: int = 5,
        max_total_prompts: int = 12,
        use_knowledge_for_concepts: bool = True,
    ) -> None:
        self.llm = llm
        self.strategy = strategy
        self.enabled = enabled
        self.max_raw_concepts = max_raw_concepts
        self.max_selected_concepts = max_selected_concepts
        self.max_total_prompts = max_total_prompts
        self.use_knowledge_for_concepts = use_knowledge_for_concepts

        self.prompt_builder = ConceptPromptBuilder(
            max_raw_concepts=max_raw_concepts,
            max_selected_concepts=max_selected_concepts,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], llm: Any) -> "ConceptGenerator":
        concept_cfg = cfg.get("concept", {})

        max_selected = int(concept_cfg.get("max_selected_concepts", 5))
        max_total = int(concept_cfg.get("max_total_prompts", max(max_selected, 12)))

        return cls(
            llm=llm,
            strategy=concept_cfg.get("strategy", "llm_concepts_knowledge"),
            enabled=bool(concept_cfg.get("enabled", True)),
            max_raw_concepts=int(concept_cfg.get("max_raw_concepts", 12)),
            max_selected_concepts=max_selected,
            max_total_prompts=max_total,
            use_knowledge_for_concepts=bool(
                concept_cfg.get("use_knowledge_for_concepts", True)
            ),
        )

    def _call_concept_llm(self, prompt: str, max_tokens: int = 200) -> str:
        if hasattr(self.llm, "generate_concepts"):
            return self.llm.generate_concepts(
                prompt=prompt,
                max_tokens=max_tokens,
            )

        return self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def _merge_with_fallbacks(
        self,
        question: str,
        selected: Sequence[str],
    ) -> List[str]:
        """
        Merge LLM-selected prompts with heuristic fallback prompts.

        The fallback prompts prevent failures such as:
            selected_prompts = ["Ball", "Racket", "Court", "Net"]
        for a motorcycle / motocross image.
        """
        fallback = heuristic_fallback_prompts(question)
        merged = list(selected) + fallback

        return clean_concepts(
            merged,
            max_items=self.max_total_prompts,
        )

    def generate_raw_concepts(self, question: str) -> tuple[List[str], str]:
        prompt = self.prompt_builder.concept_generation_prompt(question)
        raw_output = self._call_concept_llm(prompt, max_tokens=200)

        concepts = split_llm_list_output(raw_output)
        concepts = clean_concepts(
            concepts,
            max_items=self.max_raw_concepts,
        )

        return concepts, raw_output

    def expand_with_knowledge(
        self,
        question: str,
        raw_concepts: Sequence[str],
        passages: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> tuple[List[str], str]:
        prompt = self.prompt_builder.knowledge_expansion_prompt(
            question=question,
            raw_concepts=raw_concepts,
            passages=passages,
        )

        raw_output = self._call_concept_llm(prompt, max_tokens=200)

        expanded = split_llm_list_output(raw_output)

        candidate_pool = list(raw_concepts) + list(expanded)
        candidate_pool = clean_concepts(
            candidate_pool,
            max_items=self.max_raw_concepts,
        )

        return candidate_pool, raw_output

    def select_prompts(
        self,
        question: str,
        candidate_concepts: Sequence[str],
    ) -> tuple[List[str], str]:
        candidate_concepts = clean_concepts(candidate_concepts)

        if not candidate_concepts:
            return [], ""

        if len(candidate_concepts) <= self.max_selected_concepts:
            selected = clean_concepts(
                candidate_concepts,
                max_items=self.max_selected_concepts,
            )
            return selected, ""

        prompt = self.prompt_builder.prompt_selection_prompt(
            question=question,
            candidate_concepts=candidate_concepts,
        )

        raw_output = self._call_concept_llm(prompt, max_tokens=120)

        selected = split_llm_list_output(raw_output)
        selected = clean_concepts(
            selected,
            max_items=self.max_selected_concepts,
        )

        if not selected:
            selected = clean_concepts(
                candidate_concepts,
                max_items=self.max_selected_concepts,
            )

        return selected, raw_output

    def run(
        self,
        question: str,
        passages: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ConceptResult:
        """
        Run concept generation according to selected strategy.
        """
        if not self.enabled:
            prompts = question_as_prompt(
                question,
                max_selected=self.max_selected_concepts,
            )
            prompts = self._merge_with_fallbacks(question, prompts)

            return ConceptResult(
                raw_concepts=prompts,
                expanded_concepts=prompts,
                selected_prompts=prompts,
            )

        strategy = str(self.strategy).lower()

        if strategy == "question":
            prompts = question_as_prompt(
                question,
                max_selected=self.max_selected_concepts,
            )
            prompts = self._merge_with_fallbacks(question, prompts)

            return ConceptResult(
                raw_concepts=prompts,
                expanded_concepts=prompts,
                selected_prompts=prompts,
            )

        if strategy == "noun_phrases":
            prompts = extract_noun_phrases_heuristic(
                question,
                max_phrases=self.max_selected_concepts,
            )
            prompts = self._merge_with_fallbacks(question, prompts)

            return ConceptResult(
                raw_concepts=prompts,
                expanded_concepts=prompts,
                selected_prompts=prompts,
            )

        if strategy == "llm_concepts":
            raw_concepts, raw_generation_output = self.generate_raw_concepts(question)

            selected, raw_selection_output = self.select_prompts(
                question=question,
                candidate_concepts=raw_concepts,
            )

            selected = self._merge_with_fallbacks(question, selected)

            return ConceptResult(
                raw_concepts=raw_concepts,
                expanded_concepts=raw_concepts,
                selected_prompts=selected,
                raw_generation_output=raw_generation_output,
                raw_selection_output=raw_selection_output,
            )

        if strategy == "llm_concepts_knowledge":
            raw_concepts, raw_generation_output = self.generate_raw_concepts(question)

            if self.use_knowledge_for_concepts and passages:
                expanded_concepts, raw_expansion_output = self.expand_with_knowledge(
                    question=question,
                    raw_concepts=raw_concepts,
                    passages=passages,
                )
            else:
                expanded_concepts = raw_concepts
                raw_expansion_output = ""

            selected, raw_selection_output = self.select_prompts(
                question=question,
                candidate_concepts=expanded_concepts,
            )

            selected = self._merge_with_fallbacks(question, selected)

            combined_raw_output = "\n\n".join(
                part
                for part in [
                    raw_generation_output,
                    raw_expansion_output,
                ]
                if part
            )

            return ConceptResult(
                raw_concepts=raw_concepts,
                expanded_concepts=expanded_concepts,
                selected_prompts=selected,
                raw_generation_output=combined_raw_output,
                raw_selection_output=raw_selection_output,
            )

        raise ValueError(
            f"Unsupported concept strategy: {self.strategy}. "
            "Choose from: question, noun_phrases, llm_concepts, llm_concepts_knowledge."
        )


def build_concept_generator(cfg: Dict[str, Any], llm: Any) -> ConceptGenerator:
    """
    Build concept generator from config.
    """
    return ConceptGenerator.from_config(cfg=cfg, llm=llm)