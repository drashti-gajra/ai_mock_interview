"""
Generate mock interview questions from merged resume data using Ollama (Mistral).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import config

SYSTEM_PROMPT = """You are an expert technical interviewer. Given a candidate's profile, skills, and work experience, generate a comprehensive mock interview question set.

Generate EXACTLY 15 questions total (not more, not less) across 4 categories:

1. **technical** (5 questions): Test depth in their listed skills. Ask about architecture decisions, debugging scenarios, trade-offs, and tool-specific knowledge based on what they actually used. Reference specific tools and technologies from their skill list.

2. **behavioral** (4 questions): STAR-method questions tied to their real experience descriptions. Reference specific projects, achievements, or responsibilities from their resume bullet points. Ask them to describe situations they actually encountered.

3. **situational** (3 questions): Hypothetical but realistic scenarios they would face in their current role and domain. These should test problem-solving and decision-making in context.

4. **role_specific** (3 questions): Questions specific to their job title, seniority level, and industry. For a manager, ask about leadership; for an analyst, ask about methodology; match the level of depth to their experience.

Rules:
- Every question MUST reference specific skills, tools, or experiences from the provided profile
- Do NOT ask generic questions like "tell me about yourself" or "what are your strengths"
- For technical questions, go deep — ask about trade-offs, failure modes, optimization, and "why" decisions
- For behavioral questions, reference actual bullet points from their experience
- Each question must have a "difficulty" rating: "easy", "medium", or "hard"
- Include a "context" field explaining which skill or experience the question targets
- Include a "follow_up" question for deeper probing

EVALUATION CRITERIA:
For each question, you MUST provide an "evaluation" object that supports 5-dimensional scoring of candidate answers:

1. "relevance_target" (string): A concise description of the core topic the answer must address for it to be considered on-topic. Used to measure how well the response addresses the question.

2. "technical_keywords" (list of 8-10 strings): Specific domain terms, concepts, tools, or phrases that indicate technical correctness. Split into two sub-lists:
   - "core" (4-5): fundamental terms expected in any acceptable answer
   - "advanced" (4-5): deeper terms that indicate expert-level knowledge, trade-offs, or edge cases

3. "depth_indicators" (object): Signals used to judge completeness of the answer:
   - "key_concepts" (4-6 strings): specific concepts or principles the answer should cover
   - "example_expected" (boolean): whether a concrete example or scenario is required for a complete answer
   - "min_expected_points" (integer 2-5): minimum distinct points expected in a thorough answer

4. "clarity_signals" (list of 3-5 strings): logical connectors, structural phrases, or reasoning patterns that indicate a well-organized answer (e.g., "first... then...", "the trade-off is", "this caused")

5. "weights" (object): relative importance of each dimension for THIS question specifically (must sum to 1.0):
   - "relevance": float
   - "technical_accuracy": float
   - "depth": float
   - "clarity": float
   - "confidence": float

Weighting guidance by category:
- technical: emphasize technical_accuracy (0.30) and depth (0.25)
- behavioral: emphasize relevance (0.30) and depth (0.25)
- situational: emphasize relevance (0.25) and clarity (0.25)
- role_specific: balance all dimensions (~0.20 each)

- Respond ONLY with valid JSON, no markdown formatting or code blocks

JSON format:
{
  "questions": [
    {
      "id": 1,
      "category": "technical",
      "question": "...",
      "difficulty": "medium",
      "context": "Based on: [which skill/experience this targets]",
      "follow_up": "...",
      "evaluation": {
        "relevance_target": "Brief description of what an on-topic answer looks like",
        "technical_keywords": {
          "core": ["keyword1", "keyword2", "keyword3", "keyword4"],
          "advanced": ["keyword5", "keyword6", "keyword7", "keyword8"]
        },
        "depth_indicators": {
          "key_concepts": ["concept1", "concept2", "concept3", "concept4"],
          "example_expected": true,
          "min_expected_points": 3
        },
        "clarity_signals": ["phrase1", "phrase2", "phrase3"],
        "weights": {
          "relevance": 0.20,
          "technical_accuracy": 0.30,
          "depth": 0.25,
          "clarity": 0.15,
          "confidence": 0.10
        }
      }
    }
  ]
}"""


def build_user_prompt(merged_data: dict) -> str:
    profile = merged_data["candidate_profile"]
    skills = merged_data["skills"]
    experiences = merged_data["experience"]

    lines = [
        "Candidate Profile:",
        f"- Current Role: {profile['current_role']} at {profile['current_company']}",
        f"- Total Experience: {profile['total_experience_years']} years",
        f"- Domain: {profile['domain']}",
        "",
        f"Skills: {', '.join(skills)}",
        "",
        "Work Experience:",
    ]

    for i, exp in enumerate(experiences, 1):
        ongoing = " (Current)" if exp.get("is_ongoing") else ""
        lines.append(f"\n{i}. {exp['position']} at {exp['company']} ({exp['duration_display']}){ongoing}")
        if exp.get("description"):
            lines.append(f"   {exp['description']}")

    lines.append("\nGenerate the interview questions now.")
    return "\n".join(lines)


def generate_questions(merged_data: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )

    user_prompt = build_user_prompt(merged_data)

    print(f"Calling Ollama ({config.LLM_MODEL})...")
    start_time = time.time()

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )

    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.1f}s")

    raw = response.choices[0].message.content

    # Strip markdown code fences if present
    if raw.strip().startswith("```"):
        lines = raw.strip().split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    # Fix common local-model JSON issues
    raw = raw.strip()
    raw = re.sub(r',\s*([}\]])', r'\1', raw)  # remove trailing commas

    # Normalize variant key names the model might use for evaluation
    raw = re.sub(r'"difficiency_evaluation"', '"evaluation"', raw)
    raw = re.sub(r'"difficulty_evaluation"', '"evaluation"', raw)
    raw = re.sub(r'"eval"', '"evaluation"', raw)

    # Ensure the JSON is properly closed
    open_braces = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')
    raw += ']' * open_brackets + '}' * open_braces

    # Try parsing, if it fails attempt incremental repair
    try:
        questions_data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage by finding the last complete question object
        try:
            # Find all complete question blocks
            last_valid = raw.rfind('}')
            while last_valid > 0:
                attempt = raw[:last_valid + 1]
                # Close any open arrays/objects
                ob = attempt.count('{') - attempt.count('}')
                oq = attempt.count('[') - attempt.count(']')
                attempt += ']' * oq + '}' * ob
                try:
                    questions_data = json.loads(attempt)
                    print("Warning: Repaired truncated JSON response")
                    break
                except json.JSONDecodeError:
                    last_valid = raw.rfind('}', 0, last_valid)
            else:
                raise json.JSONDecodeError("Could not repair", raw, 0)
        except json.JSONDecodeError:
            print("Error: Failed to parse LLM response as JSON.")
            print("Raw response (first 500 chars):")
            print(raw[:500])
            sys.exit(1)

    # Enforce max 15 questions
    questions_list = questions_data.get("questions", [])
    if len(questions_list) > 15:
        questions_list = questions_list[:15]
        print(f"Trimmed to 15 questions (model generated {len(questions_data.get('questions', []))})")

    profile = merged_data["candidate_profile"]
    result = {
        "metadata": {
            "candidate": f"{profile['current_role']} at {profile['current_company']}",
            "domain": profile["domain"],
            "total_experience_years": profile["total_experience_years"],
            "total_questions": len(questions_list),
            "generated_at": datetime.now().isoformat(),
            "generation_time_seconds": round(elapsed, 1),
            "model": config.LLM_MODEL,
        },
        "questions": questions_list,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate mock interview questions using Ollama (Mistral)")
    parser.add_argument("--input", default="merged_output.json", help="Path to merged JSON")
    parser.add_argument("--skills", default=None, help="Path to skills JSON (auto-merges with --experience)")
    parser.add_argument("--experience", default=None, help="Path to experience JSON (auto-merges with --skills)")
    parser.add_argument("--output", default="questions_output.json", help="Output path for questions JSON")
    args = parser.parse_args()

    # Auto-merge if both --skills and --experience are provided
    if args.skills and args.experience:
        from merge_outputs import merge
        merged_data = merge(args.skills, args.experience, "merged_output.json")
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            print("Run merge_outputs.py first, or provide --skills and --experience flags.")
            sys.exit(1)
        with open(args.input, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

    result = generate_questions(merged_data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {result['metadata']['total_questions']} questions -> {args.output}")

    # Print summary
    categories = {}
    for q in result["questions"]:
        cat = q.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} questions")


if __name__ == "__main__":
    main()
