import re
from collections.abc import Iterable

import torch
from tqdm import tqdm
from transformers import pipeline

_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCTUATION = " .,;:()[]{}\"'"

GENERIC_TERMS = frozenset({
    "patients", "patient", "study", "studies", "results", "result", "group", "groups",
    "analysis", "data", "method", "methods", "treatment", "years", "year", "age", "women",
    "men", "children", "cases", "case", "control", "controls", "risk", "effect", "effects",
    "outcome", "outcomes", "purpose", "objective", "background", "conclusion", "conclusions",
    "increase", "decrease", "level", "levels", "rate", "rates", "factor", "factors",
    "association", "participants", "subjects", "sample", "population", "evidence", "use",
    "time", "period", "number", "total", "mean", "median", "significant", "difference",
})

# Labels from the MACCROBAT scheme that describe measurements, demographics or
# narrative context rather than medical concepts worth linking in the graph.
EXCLUDED_LABELS = frozenset({
    "Activity", "Administration", "Age", "Area", "Color", "Coreference", "Date",
    "Detailed_description", "Distance", "Dosage", "Duration", "Family_history",
    "Frequency", "Height", "History", "Lab_value", "Mass", "Nonbiological_location",
    "Occupation", "Other_entity", "Other_event", "Outcome", "Personal_background",
    "Qualitative_concept", "Quantitative_concept", "Severity", "Sex", "Shape",
    "Subject", "Texture", "Time", "Volume", "Weight",
})


def normalize(text: str) -> str | None:
    name = _WHITESPACE.sub(" ", text.lower()).strip(_EDGE_PUNCTUATION)
    if len(name) < 3 or name.isdigit() or name in GENERIC_TERMS:
        return None
    return name


class EntityExtractor:
    def __init__(self, model_name: str = "d4data/biomedical-ner-all", batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.ner = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="first",
            stride=128,  # sliding window so abstracts beyond 512 tokens are fully tagged
            device=0 if torch.cuda.is_available() else -1,
        )

    def _entities(self, text: str, spans: list[dict]) -> set[str]:
        return {
            name
            for span in spans
            if span["entity_group"] not in EXCLUDED_LABELS
            and (name := normalize(text[span["start"]:span["end"]]))
        }

    def extract(self, texts: Iterable[str], show_progress: bool = False) -> list[set[str]]:
        texts = list(texts)
        # A generator makes the pipeline stream batches instead of returning one list,
        # which is what lets tqdm report progress.
        outputs = self.ner((t for t in texts), batch_size=self.batch_size)
        if show_progress:
            outputs = tqdm(outputs, total=len(texts), desc="NER")
        return [self._entities(text, spans) for text, spans in zip(texts, outputs)]
