import re

import numpy as np
from sentence_transformers import CrossEncoder

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_CLAIM_WORDS = 5


def split_claims(rationale: str) -> list[str]:
    # Very short fragments ("Answer: yes.") are verdicts, not factual claims.
    return [
        s.strip()
        for s in _SENTENCE_SPLIT.split(rationale)
        if len(s.split()) >= MIN_CLAIM_WORDS
    ]


class FaithfulnessScorer:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.5,
        batch_size: int = 32,
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=512)
        self.threshold = threshold
        self.batch_size = batch_size
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.entailment_index = labels["entailment"]

    def score(self, rationales: list[str], contexts: list[list[str]]) -> list[float | None]:
        """Share of each rationale's claims entailed by at least one retrieved passage.

        Returns None for rationales without checkable claims or without context.
        """
        pairs, owners = [], []
        claims_per_row = [split_claims(r) for r in rationales]
        for row, (claims, passages) in enumerate(zip(claims_per_row, contexts)):
            for claim_idx, claim in enumerate(claims):
                for passage in passages:
                    pairs.append((passage, claim))
                    owners.append((row, claim_idx))

        if not pairs:
            return [None] * len(rationales)

        probs = self.model.predict(
            pairs, batch_size=self.batch_size, apply_softmax=True, show_progress_bar=True
        )
        entailment = np.asarray(probs)[:, self.entailment_index]

        best: dict[tuple[int, int], float] = {}
        for owner, p in zip(owners, entailment):
            best[owner] = max(best.get(owner, 0.0), float(p))

        scores: list[float | None] = []
        for row, (claims, passages) in enumerate(zip(claims_per_row, contexts)):
            if not claims or not passages:
                scores.append(None)
                continue
            supported = sum(best[(row, i)] >= self.threshold for i in range(len(claims)))
            scores.append(supported / len(claims))
        return scores
