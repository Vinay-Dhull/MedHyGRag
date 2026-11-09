from src.types import Messages

_ANSWER_POLICY = (
    "Answer yes or no whenever the evidence or established medical knowledge supports a "
    "direction. Answer maybe only when the findings are genuinely mixed or inconclusive."
)

SYSTEM_PROMPTS = {
    "v1": (
        "You are a biomedical research assistant. Answer the research question "
        "with exactly one lowercase word: yes, no, or maybe."
    ),
    "v2": (
        "You are a biomedical research assistant. Reply with exactly one lowercase word: "
        f"yes, no, or maybe. {_ANSWER_POLICY}"
    ),
    "v3": (
        "You are a biomedical research assistant. Reason briefly, in at most three "
        "sentences, about what the available information shows, then state whether the "
        f"answer is yes, no, or maybe. {_ANSWER_POLICY}"
    ),
}

REASONING_PROMPTS = frozenset({"v3"})
FINAL_ANSWER_CUE = "\n\nFinal answer:"


def _question_block(question: str, prompt_version: str) -> str:
    if prompt_version in REASONING_PROMPTS:
        return f"Question: {question}"
    return f"Question: {question}\nAnswer:"


def plain_messages(question: str, prompt_version: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": _question_block(question, prompt_version)},
    ]


def rag_messages(
    question: str,
    contexts: list[str],
    prompt_version: str,
    facts: list[str] | None = None,
) -> Messages:
    sections = ["Evidence:\n" + "\n\n".join(f"[{i}] {t}" for i, t in enumerate(contexts, start=1))]
    if facts:
        sections.append("Knowledge graph relations:\n" + "\n".join(f"- {f}" for f in facts))
    sections.append(
        "Using the information above, answer the question.\n"
        + _question_block(question, prompt_version)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def with_final_answer_cue(messages: Messages, rationale: str) -> Messages:
    return [*messages, {"role": "assistant", "content": rationale.strip() + FINAL_ANSWER_CUE}]
