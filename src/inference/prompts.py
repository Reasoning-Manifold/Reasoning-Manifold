
def format_mmlu_prompt(question: str, choices: list[str] | None) -> str:
    if not choices:
        return question
    parts = [question, "", "Options:"]
    for idx, choice in enumerate(choices):
        parts.append(f"{chr(65 + idx)}. {choice}")
    parts.extend(["", "Please select the correct answer."])
    return "\n".join(parts)


def format_gpqa_prompt(question: str) -> str:
    return f"Question: {question}\n\nPlease answer the question and put the final answer in \\boxed{{}}."
