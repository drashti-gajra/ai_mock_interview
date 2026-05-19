"""
Multi-dimensional feedback evaluator for mock interview responses.

Implements the 5-dimension scoring model from the paper:
  1. Relevance        — embedding-based cosine similarity (Sim(Q, R) = Q·R / ||Q||||R||)
  2. Technical Accuracy — keyword matching + LLM contextual validation
  3. Depth & Completeness — response length, key concept coverage, example presence
  4. Clarity & Communication — filler-word detection, logical-connector presence
  5. Confidence       — text-based proxy (fluency, directness, hedging language)

Overall score:
  Score = w1*R + w2*A + w3*D + w4*C + w5*Conf   (weights per question, default equal)

All individual dimension scores are on a 1–5 scale.
"""

import math
import re
import time
from typing import Optional

import config

# ---------------------------------------------------------------------------
# Filler / hedging word lists
# ---------------------------------------------------------------------------

FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "basically", "literally", "actually",
    "you know", "i mean", "sort of", "kind of", "i guess", "i think",
    "probably", "maybe", "perhaps",
}

HEDGING_PHRASES = [
    r"\bi think\b", r"\bi guess\b", r"\bprobably\b", r"\bmaybe\b",
    r"\bnot sure\b", r"\bkind of\b", r"\bsort of\b", r"\bi believe\b",
    r"\bit might be\b", r"\bit could be\b",
]

LOGICAL_CONNECTORS = [
    r"\bfirst(ly)?\b", r"\bsecond(ly)?\b", r"\bthird(ly)?\b",
    r"\bfinally\b", r"\bmoreover\b", r"\bfurthermore\b", r"\bin addition\b",
    r"\bhowever\b", r"\btherefore\b", r"\bas a result\b", r"\bbecause\b",
    r"\bthe reason\b", r"\bthis (caused|led|resulted)\b",
    r"\bthe trade-?off\b", r"\bon the other hand\b", r"\bfor example\b",
    r"\bfor instance\b", r"\bspecifically\b", r"\bin contrast\b",
]

EXAMPLE_SIGNALS = [
    r"\bfor example\b", r"\bfor instance\b", r"\bsuch as\b",
    r"\bin my (previous |current )?(role|project|team|company)\b",
    r"\bwe (built|implemented|used|deployed|migrated|designed)\b",
    r"\bi (built|implemented|used|deployed|created|developed)\b",
    r"\bone time\b", r"\bonce (we|i)\b",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r'[.!?]+', text.strip())))


def _count_pattern_hits(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if re.search(p, lowered))


def _filler_rate(text: str) -> float:
    """Fraction of words that are filler words (0.0 – 1.0)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in FILLER_WORDS)
    # Also catch two-word fillers
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    hits += sum(1 for b in bigrams if b in FILLER_WORDS)
    return hits / len(words)


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """
    Lightweight TF-IDF cosine similarity — no external dependencies.
    Returns a float in [0, 1].
    """
    def tokenize(t: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", t.lower())

    STOPWORDS = {
        "the", "a", "an", "is", "in", "on", "at", "of", "to", "and",
        "or", "for", "with", "as", "by", "be", "it", "its", "this",
        "that", "are", "was", "were", "have", "has", "had", "you", "your",
        "can", "do", "did", "will", "would", "could", "should", "may",
        "from", "into", "about", "their", "they", "them", "than",
    }

    tokens_a = [t for t in tokenize(text_a) if t not in STOPWORDS]
    tokens_b = [t for t in tokenize(text_b) if t not in STOPWORDS]

    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0

    def tf(tokens: list[str]) -> dict[str, float]:
        freq: dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = len(tokens) or 1
        return {k: v / total for k, v in freq.items()}

    tf_a, tf_b = tf(tokens_a), tf(tokens_b)

    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    norm_a = math.sqrt(sum(v * v for v in tf_a.values())) or 1e-9
    norm_b = math.sqrt(sum(v * v for v in tf_b.values())) or 1e-9

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Dimension scorers  (all return float in [1, 5])
# ---------------------------------------------------------------------------


def score_relevance(question: str, answer: str) -> dict:
    """
    Relevance: Sim(Q, R) = Q·R / ||Q||||R||   (paper eq. 1)
    Maps cosine similarity [0, 1] → score [1, 5].
    """
    sim = _tfidf_cosine(question, answer)
    score = 1 + sim * 4  # linear mapping: 0→1, 1→5
    return {
        "score": round(score, 2),
        "cosine_similarity": round(sim, 3),
        "notes": "Measures how well the answer addresses the question topic.",
    }


def score_technical_accuracy(answer: str, technical_keywords: dict) -> dict:
    """
    Technical Accuracy: keyword matching against core and advanced keyword lists.
    core hits → base accuracy; advanced hits → bonus.
    Score range [1, 5].
    """
    core_kws = [k.lower() for k in technical_keywords.get("core", [])]
    adv_kws = [k.lower() for k in technical_keywords.get("advanced", [])]
    answer_lower = answer.lower()

    core_hits = sum(1 for k in core_kws if k in answer_lower)
    adv_hits = sum(1 for k in adv_kws if k in answer_lower)

    core_ratio = core_hits / len(core_kws) if core_kws else 0
    adv_ratio = adv_hits / len(adv_kws) if adv_kws else 0

    # Weighted: core accounts for 60%, advanced for 40%
    combined = 0.6 * core_ratio + 0.4 * adv_ratio
    score = 1 + combined * 4

    return {
        "score": round(score, 2),
        "core_keywords_hit": core_hits,
        "core_keywords_total": len(core_kws),
        "advanced_keywords_hit": adv_hits,
        "advanced_keywords_total": len(adv_kws),
        "notes": f"Core coverage {core_hits}/{len(core_kws)}, Advanced {adv_hits}/{len(adv_kws)}.",
    }


def score_depth(answer: str, depth_indicators: dict) -> dict:
    """
    Depth & Completeness: key concept coverage + example presence + response length.
    Score range [1, 5].
    """
    key_concepts = [c.lower() for c in depth_indicators.get("key_concepts", [])]
    example_expected: bool = depth_indicators.get("example_expected", False)
    min_points: int = depth_indicators.get("min_expected_points", 3)

    answer_lower = answer.lower()
    wc = _word_count(answer)

    # Concept coverage
    concept_hits = sum(1 for c in key_concepts if c in answer_lower)
    concept_ratio = concept_hits / len(key_concepts) if key_concepts else 1.0

    # Example presence
    has_example = _count_pattern_hits(answer, EXAMPLE_SIGNALS) > 0
    example_score = 1.0 if (not example_expected or has_example) else 0.5

    # Length proxy for "minimum points" — heuristic: 40 words per expected point
    length_ratio = min(1.0, wc / max(1, min_points * 40))

    combined = (0.5 * concept_ratio + 0.3 * length_ratio + 0.2 * example_score)
    score = 1 + combined * 4

    return {
        "score": round(score, 2),
        "key_concepts_hit": concept_hits,
        "key_concepts_total": len(key_concepts),
        "word_count": wc,
        "example_detected": has_example,
        "notes": (
            f"Covered {concept_hits}/{len(key_concepts)} key concepts. "
            f"Word count: {wc}. Example {'found' if has_example else 'missing'}."
        ),
    }


def score_clarity(answer: str, clarity_signals: Optional[list] = None) -> dict:
    """
    Clarity & Communication: logical flow (connector presence) minus filler-word penalty.
    Also rewards question-specific clarity signals from evaluation metadata.
    Score range [1, 5].
    """
    clarity_signals = clarity_signals or []

    filler_rate = _filler_rate(answer)
    connector_hits = _count_pattern_hits(answer, LOGICAL_CONNECTORS)
    wc = _word_count(answer)
    sent_count = _sentence_count(answer)
    avg_sentence_len = wc / sent_count  # readability proxy

    # Connector density: connectors per 100 words
    connector_density = (connector_hits / max(wc, 1)) * 100
    connector_score = min(1.0, connector_density / 5)  # 5 connectors/100w → full score

    # Filler penalty: 0 fillers → no penalty; >10% fillers → −0.5 contribution
    filler_penalty = min(0.5, filler_rate * 5)

    # Question-specific clarity signals
    custom_hits = sum(
        1 for s in clarity_signals if s.lower() in answer.lower()
    )
    custom_score = min(1.0, custom_hits / max(len(clarity_signals), 1))

    # Sentence length penalty: very long avg sentences hurt clarity
    length_penalty = 0.1 if avg_sentence_len > 35 else 0.0

    combined = (
        0.40 * connector_score
        + 0.30 * custom_score
        - 0.20 * filler_penalty
        - length_penalty
    )
    combined = max(0.0, min(1.0, combined + 0.30))  # baseline shift so 0 → ~1.3/5
    score = 1 + combined * 4

    return {
        "score": round(score, 2),
        "filler_rate": round(filler_rate, 3),
        "logical_connectors_found": connector_hits,
        "clarity_signals_hit": custom_hits,
        "avg_sentence_length_words": round(avg_sentence_len, 1),
        "notes": (
            f"Filler rate {filler_rate:.1%}. "
            f"Logical connectors: {connector_hits}. "
            f"Custom clarity signals: {custom_hits}/{len(clarity_signals)}."
        ),
    }


def score_confidence(
    answer: str,
    response_delay_ms: Optional[float] = None,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
) -> dict:
    """
    Confidence: Conf = α(1 − Delay) + β(SpeechRate) − γ(Hesitation)   (paper eq. 2)

    In text-only mode:
      • Delay: if response_delay_ms is provided, normalise against 3000 ms max.
               Otherwise estimated as 0 (unknown → neutral).
      • SpeechRate: word count as proxy for fluency (100–200 wpm is conversational).
      • Hesitation: hedging-phrase density.

    Score range [1, 5].
    """
    wc = _word_count(answer)
    hedging_hits = _count_pattern_hits(answer, HEDGING_PHRASES)
    hedging_rate = hedging_hits / max(_sentence_count(answer), 1)

    # Normalise delay: 0 ms → 1.0 (no delay), ≥3000 ms → 0.0
    if response_delay_ms is not None:
        delay_norm = max(0.0, 1.0 - response_delay_ms / 3000.0)
    else:
        delay_norm = 0.5  # neutral when no timing data

    # Speech-rate proxy: assume ~130 wpm for conversational pace
    # We don't know duration, so use word count relative to expected
    # A well-formed interview answer is 100–250 words.
    speech_rate_norm = min(1.0, wc / 150)

    # Hesitation penalty
    hesitation_norm = min(1.0, hedging_rate)

    raw = alpha * delay_norm + beta * speech_rate_norm - gamma * hesitation_norm
    raw = max(0.0, min(1.0, raw))
    score = 1 + raw * 4

    return {
        "score": round(score, 2),
        "response_delay_ms": response_delay_ms,
        "word_count": wc,
        "hedging_phrases_found": hedging_hits,
        "notes": (
            f"Delay component: {delay_norm:.2f}. "
            f"Fluency (word count proxy): {wc} words. "
            f"Hedging phrases: {hedging_hits}."
        ),
    }


# ---------------------------------------------------------------------------
# LLM-assisted technical accuracy boost (optional)
# ---------------------------------------------------------------------------


def llm_validate_technical_accuracy(
    question: str, answer: str, context: str
) -> Optional[float]:
    """
    Use Mistral via Ollama to cross-validate technical correctness.
    Returns a float in [1, 5] or None if the call fails.
    Only called when keyword matching alone is insufficient.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

        prompt = (
            f"You are a strict technical interviewer grading a candidate's answer.\n\n"
            f"Question: {question}\n"
            f"Context: {context}\n"
            f"Candidate's answer: {answer}\n\n"
            f"Rate the TECHNICAL ACCURACY of the answer on a scale of 1 to 5:\n"
            f"1 = completely wrong or off-topic\n"
            f"2 = partially correct with significant gaps\n"
            f"3 = mostly correct, minor inaccuracies\n"
            f"4 = accurate with good depth\n"
            f"5 = expert-level correctness and nuance\n\n"
            f"Respond with ONLY a single integer between 1 and 5."
        )

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        val = float(re.search(r"[1-5]", raw).group())
        return max(1.0, min(5.0, val))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_response(
    question_obj: dict,
    answer: str,
    response_delay_ms: Optional[float] = None,
    use_llm_validation: bool = False,
) -> dict:
    """
    Evaluate a candidate's answer against the 5-dimension model.

    Parameters
    ----------
    question_obj : dict
        A single question entry from questions_output.json (must include 'evaluation').
    answer : str
        The candidate's free-text answer.
    response_delay_ms : float | None
        Time (ms) from question end to answer start — used for confidence scoring.
        Pass None when no timing data is available.
    use_llm_validation : bool
        Whether to call Ollama/Mistral for technical accuracy validation.
        Adds ~1–3 s per call; disabled by default for speed.

    Returns
    -------
    dict with keys:
        scores        — per-dimension float [1, 5]
        overall       — weighted composite score [1, 5]
        breakdown     — detailed sub-metrics per dimension
        weights_used  — the weights applied
    """
    question_text = question_obj.get("question", "")
    context = question_obj.get("context", "")
    eval_meta = question_obj.get("evaluation", {})

    # --- Extract evaluation metadata ---
    relevance_target: str = eval_meta.get("relevance_target", question_text)
    technical_keywords: dict = eval_meta.get("technical_keywords", {"core": [], "advanced": []})
    depth_indicators: dict = eval_meta.get("depth_indicators", {})
    clarity_signals: list = eval_meta.get("clarity_signals", [])
    weights: dict = eval_meta.get("weights", {
        "relevance": 0.20,
        "technical_accuracy": 0.25,
        "depth": 0.25,
        "clarity": 0.15,
        "confidence": 0.15,
    })

    # --- Score each dimension ---
    rel = score_relevance(relevance_target, answer)
    tec = score_technical_accuracy(answer, technical_keywords)
    dep = score_depth(answer, depth_indicators)
    cla = score_clarity(answer, clarity_signals)
    con = score_confidence(answer, response_delay_ms)

    # Optional LLM boost for technical accuracy
    if use_llm_validation and technical_keywords.get("core"):
        llm_score = llm_validate_technical_accuracy(question_text, answer, context)
        if llm_score is not None:
            # Blend keyword score (40%) with LLM score (60%)
            blended = 0.4 * tec["score"] + 0.6 * llm_score
            tec["score"] = round(blended, 2)
            tec["llm_validation_score"] = llm_score
            tec["notes"] += f" LLM validation: {llm_score}/5."

    # --- Compute weighted overall score ---
    w_r = weights.get("relevance", 0.20)
    w_a = weights.get("technical_accuracy", 0.25)
    w_d = weights.get("depth", 0.25)
    w_c = weights.get("clarity", 0.15)
    w_conf = weights.get("confidence", 0.15)

    overall = (
        w_r * rel["score"]
        + w_a * tec["score"]
        + w_d * dep["score"]
        + w_c * cla["score"]
        + w_conf * con["score"]
    )

    return {
        "question_id": question_obj.get("id"),
        "question": question_text,
        "answer_preview": answer[:120] + ("..." if len(answer) > 120 else ""),
        "scores": {
            "relevance": rel["score"],
            "technical_accuracy": tec["score"],
            "depth": dep["score"],
            "clarity": cla["score"],
            "confidence": con["score"],
        },
        "overall": round(overall, 2),
        "weights_used": weights,
        "breakdown": {
            "relevance": rel,
            "technical_accuracy": tec,
            "depth": dep,
            "clarity": cla,
            "confidence": con,
        },
    }


def evaluate_session(
    questions: list[dict],
    answers: list[str],
    response_delays_ms: Optional[list[Optional[float]]] = None,
    use_llm_validation: bool = False,
) -> dict:
    """
    Evaluate all answers in a session and produce a per-question score matrix
    plus aggregate performance profile.

    Parameters
    ----------
    questions : list of question dicts from questions_output.json
    answers   : list of candidate answer strings (same order as questions)
    response_delays_ms : optional list of delay values (ms), one per answer
    use_llm_validation : whether to enable LLM technical accuracy check

    Returns
    -------
    dict with:
        per_question  — list of evaluate_response results
        aggregate     — mean scores per dimension and overall
        performance_profile — dimension-level strengths/weaknesses summary
    """
    if response_delays_ms is None:
        response_delays_ms = [None] * len(answers)

    per_question = []
    for q, a, delay in zip(questions, answers, response_delays_ms):
        result = evaluate_response(
            q, a,
            response_delay_ms=delay,
            use_llm_validation=use_llm_validation,
        )
        per_question.append(result)

    # Aggregate
    dimensions = ["relevance", "technical_accuracy", "depth", "clarity", "confidence"]
    agg: dict[str, float] = {}
    for dim in dimensions:
        scores = [r["scores"][dim] for r in per_question]
        agg[dim] = round(sum(scores) / len(scores), 2) if scores else 0.0
    agg["overall"] = round(sum(r["overall"] for r in per_question) / max(len(per_question), 1), 2)

    # Performance profile
    def _label(score: float) -> str:
        if score >= 4.5:
            return "Excellent"
        if score >= 3.5:
            return "Good"
        if score >= 2.5:
            return "Fair"
        return "Needs Improvement"

    performance_profile = {dim: _label(agg[dim]) for dim in dimensions}
    weakest = min(dimensions, key=lambda d: agg[d])
    strongest = max(dimensions, key=lambda d: agg[d])
    performance_profile["strengths"] = strongest
    performance_profile["areas_for_improvement"] = weakest

    return {
        "per_question": per_question,
        "aggregate": agg,
        "performance_profile": performance_profile,
    }


# ---------------------------------------------------------------------------
# CLI — evaluate a single answer interactively
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a candidate's answer against the 5-dimension model.")
    parser.add_argument("--questions", default="questions_output.json", help="Path to questions JSON")
    parser.add_argument("--question_id", type=int, default=1, help="Question ID to evaluate")
    parser.add_argument("--answer", type=str, required=True, help="Candidate's answer text")
    parser.add_argument("--delay_ms", type=float, default=None, help="Response delay in ms (optional)")
    parser.add_argument("--llm", action="store_true", help="Enable LLM technical accuracy validation")
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    q_obj = next((q for q in questions if q["id"] == args.question_id), None)
    if q_obj is None:
        print(f"Question ID {args.question_id} not found.")
        raise SystemExit(1)

    print(f"\nQuestion: {q_obj['question']}\n")
    print(f"Answer  : {args.answer}\n")

    result = evaluate_response(
        q_obj,
        args.answer,
        response_delay_ms=args.delay_ms,
        use_llm_validation=args.llm,
    )

    print("=" * 60)
    print("5-DIMENSION FEEDBACK SCORES")
    print("=" * 60)
    for dim, score in result["scores"].items():
        bar = "#" * int(score * 4)
        print(f"  {dim:<22} {score:>4.1f}/5  {bar}")
    print("-" * 60)
    print(f"  {'OVERALL':<22} {result['overall']:>4.1f}/5")
    print("=" * 60)

    print("\nDetailed Breakdown:")
    for dim, details in result["breakdown"].items():
        print(f"\n  [{dim.upper()}]")
        for k, v in details.items():
            if k != "score":
                print(f"    {k}: {v}")
