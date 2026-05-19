"""
Merge skills_output.json and experience_output.json into a single merged_output.json.
"""

import argparse
import json
import os
import sys

from domain_detector import detect_domain


def merge(skills_path: str, experience_path: str, output_path: str) -> dict:
    if not os.path.exists(skills_path):
        print(f"Error: Skills file not found: {skills_path}")
        sys.exit(1)
    if not os.path.exists(experience_path):
        print(f"Error: Experience file not found: {experience_path}")
        sys.exit(1)

    with open(skills_path, "r", encoding="utf-8") as f:
        skills_data = json.load(f)
    with open(experience_path, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    skills = skills_data.get("skills", [])
    experiences = exp_data.get("experience", [])

    # Compute total experience
    total_years = round(sum(e.get("duration_years", 0) for e in experiences), 1)

    # Find current role
    current = next((e for e in experiences if e.get("is_ongoing")), None)
    if current is None and experiences:
        current = experiences[0]

    # Detect domain from descriptions + skills text
    all_text = " ".join(skills)
    for e in experiences:
        all_text += " " + e.get("description", "") + " " + e.get("position", "")
    domain, _, _ = detect_domain(all_text)

    # Build candidate profile
    candidate_profile = {
        "domain": domain,
        "total_experience_years": total_years,
        "current_role": current["position"] if current else "",
        "current_company": current["company"].rstrip(",").strip() if current else "",
    }

    # Build clean experience list
    clean_experiences = []
    for e in experiences:
        clean_experiences.append({
            "position": e["position"],
            "company": e["company"].rstrip(",").strip(),
            "duration_display": e.get("duration_display", ""),
            "is_ongoing": e.get("is_ongoing", False),
            "description": e.get("description", ""),
        })

    merged = {
        "candidate_profile": candidate_profile,
        "skills": skills,
        "experience": clean_experiences,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Merged output saved to: {output_path}")
    print(f"  Domain: {domain}")
    print(f"  Skills: {len(skills)}")
    print(f"  Experience entries: {len(clean_experiences)}")
    print(f"  Total experience: {total_years} years")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge skills and experience outputs")
    parser.add_argument("--skills", default="skills_output.json", help="Path to skills JSON")
    parser.add_argument("--experience", default="experience_output.json", help="Path to experience JSON")
    parser.add_argument("--output", default="merged_output.json", help="Output path for merged JSON")
    args = parser.parse_args()

    merge(args.skills, args.experience, args.output)


if __name__ == "__main__":
    main()
