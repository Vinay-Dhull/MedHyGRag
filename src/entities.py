import re
from collections.abc import Iterable

import spacy
from tqdm import tqdm

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


def normalize(text: str) -> str | None:
    name = _WHITESPACE.sub(" ", text.lower()).strip(_EDGE_PUNCTUATION)
    if len(name) < 3 or name.isdigit() or name in GENERIC_TERMS:
        return None
    return name


class EntityExtractor:
    def __init__(self, model_name: str = "en_core_sci_sm") -> None:
        self.nlp = spacy.load(
            model_name, exclude=["parser", "tagger", "attribute_ruler", "lemmatizer"]
        )

    def extract(
        self, texts: Iterable[str], batch_size: int = 256, show_progress: bool = False
    ) -> list[set[str]]:
        texts = list(texts)
        docs = self.nlp.pipe(texts, batch_size=batch_size)
        if show_progress:
            docs = tqdm(docs, total=len(texts), desc="NER")
        return [{name for ent in doc.ents if (name := normalize(ent.text))} for doc in docs]
