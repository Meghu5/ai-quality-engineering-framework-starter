from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIEvalThresholds:
    ragas_faithfulness: float = 0.80
    ragas_answer_relevance: float = 0.80
    ragas_context_relevance: float = 0.80
    deepeval_answer_relevancy: float = 0.80
    deepeval_faithfulness: float = 0.80
    deepeval_contextual_relevancy: float = 0.80
    promptfoo_pass_rate: float = 0.90


DEFAULT_AI_EVAL_THRESHOLDS = AIEvalThresholds()
