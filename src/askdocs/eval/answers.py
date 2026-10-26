"""End-to-end answer quality, scored by an LLM judge.

Per question we record:

* correct     - does the answer agree with the reference answer? (yes=1, partial=0.5, no=0)
* faithful    - is every claim supported by the retrieved sources? (catches hallucination)
* cites_evidence - did the answer cite at least one source that contains the gold evidence?
* refused     - did the system say it couldn't find the answer?
"""

from __future__ import annotations

import numpy as np

from ..answer import answer_question, format_sources
from ..llm import LLM, parse_json
from ..retrieve import Retriever
from .relevance import is_relevant

JUDGE_SYSTEM = "You are a strict grader of answers produced by a documentation Q&A system. You reply in JSON."

JUDGE_PROMPT = """Question: {question}

Reference answer (ground truth): {reference}

Sources the system was given:
<sources>
{sources}
</sources>

System answer:
<answer>
{answer}
</answer>

Grade the system answer.
- "correct": "yes" if it gives the same essential information as the reference answer, "partial" if it is incomplete but not wrong, "no" if it is wrong, missing, or a refusal.
- "faithful": true if every factual claim in the system answer is supported by the sources (a refusal counts as faithful); false if it states anything the sources do not support.
- "reason": one short sentence.

Return JSON: {{"correct": "yes|partial|no", "faithful": true|false, "reason": "..."}}"""

CORRECT_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def judge(judge_llm: LLM, item: dict, ans) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        reference=item["answer"],
        sources=format_sources(ans.hits),
        answer=ans.text,
    )
    try:
        v = parse_json(judge_llm.complete(JUDGE_SYSTEM, prompt, json_mode=True, max_tokens=300))
    except Exception as e:
        return {"correct": "no", "faithful": False, "reason": f"judge error: {e}", "judge_error": True}
    correct = str(v.get("correct", "no")).lower().strip()
    faithful = v.get("faithful")
    if isinstance(faithful, str):
        faithful = faithful.lower().strip() == "true"
    return {
        "correct": correct if correct in CORRECT_SCORE else "no",
        "faithful": bool(faithful),
        "reason": str(v.get("reason", "")),
    }


def evaluate_answers(
    dataset: list[dict],
    retriever: Retriever,
    llm: LLM,
    judge_llm: LLM,
    mode: str,
    k: int = 5,
    log=print,
) -> tuple[dict, list[dict]]:
    # Generate everything first, then judge: a local Ollama server often keeps one
    # model in memory, and alternating generator/judge would reload it every call.
    answers = []
    for i, item in enumerate(dataset, 1):
        answers.append(answer_question(item["question"], retriever, llm, k=k, mode=mode))
        log(f"[{mode} answer {i}/{len(dataset)}]")

    records: list[dict] = []
    for i, (item, ans) in enumerate(zip(dataset, answers), 1):
        verdict = judge(judge_llm, item, ans)
        cites_evidence = any(is_relevant(ans.hits[n - 1].chunk, item) for n in ans.cited)
        records.append(
            {
                "id": item["id"],
                "mode": mode,
                "question": item["question"],
                "answer": ans.text,
                "retrieved_evidence": any(is_relevant(h.chunk, item) for h in ans.hits),
                "cites_evidence": cites_evidence,
                "refused": ans.refused,
                "generation_ms": ans.generation_ms,
                **verdict,
            }
        )
        log(f"[{mode} {i}/{len(dataset)}] correct={verdict['correct']} faithful={verdict['faithful']}")

    n = len(records)
    gen = np.array([r["generation_ms"] for r in records])
    summary = {
        "mode": mode,
        "n": n,
        "generator": llm.name,
        "judge": judge_llm.name,
        "correctness": sum(CORRECT_SCORE[r["correct"]] for r in records) / n,
        "faithfulness": sum(r["faithful"] for r in records) / n,
        "evidence_in_context": sum(r["retrieved_evidence"] for r in records) / n,
        "cites_evidence": sum(r["cites_evidence"] for r in records) / n,
        "refusal_rate": sum(r["refused"] for r in records) / n,
        "generation_p50_ms": float(np.percentile(gen, 50)),
    }
    return summary, records


def answers_table(rows: list[dict]) -> str:
    head = "| Retrieval | Correctness | Faithfulness | Evidence in context | Cites evidence | Refusals | Gen p50 s |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['correctness']:.1%} | {r['faithfulness']:.1%} | {r['evidence_in_context']:.1%} "
            f"| {r['cites_evidence']:.1%} | {r['refusal_rate']:.1%} | {r['generation_p50_ms'] / 1000:.1f} |"
        )
    return "\n".join(lines)
