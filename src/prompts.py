SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the research question "
    "with exactly one lowercase word: yes, no, or maybe."
)

Messages = list[dict[str, str]]


def plain_messages(question: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]


def rag_messages(question: str, contexts: list[str]) -> Messages:
    evidence = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Evidence:\n{evidence}\n\n"
                "Using only the evidence above, answer the question.\n"
                f"Question: {question}\nAnswer:"
            ),
        },
    ]
