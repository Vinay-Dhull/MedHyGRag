from src.types import Messages

SYSTEM_PROMPTS = {
    "v1": (
        "You are a biomedical research assistant. Answer the research question "
        "with exactly one lowercase word: yes, no, or maybe."
    ),
    "v2": (
        "You are a biomedical research assistant. Reply with exactly one lowercase word: "
        "yes, no, or maybe. Answer yes or no whenever the evidence or established medical "
        "knowledge supports a direction. Answer maybe only when the findings are genuinely "
        "mixed or inconclusive."
    ),
}


def plain_messages(question: str, prompt_version: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
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
    sections.append(f"Using the information above, answer the question.\nQuestion: {question}\nAnswer:")
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[prompt_version]},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
