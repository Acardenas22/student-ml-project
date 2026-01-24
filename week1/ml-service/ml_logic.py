from typing import Tuple, List
from schemas import SuggestReq, Suggestion

# Mapping table: (article, violation_code) -> suggestion metadata
ML_MAPPINGS = {
    ("310", "ampacity_mismatch"): {
        "fix_template": "Increase conductor size one AWG step and re-evaluate ampacity",
        "nec_refs": ["310.15(B)", "110.14(C)"],
        "base_priority": 0.7,
    },
    ("250", "egc_undersized"): {
        "fix_template": "Increase equipment grounding conductor to next standard size",
        "nec_refs": ["250.122"],
        "base_priority": 0.6,
    },
}


def generate_suggestions(req: SuggestReq) -> Tuple[List[Suggestion], List[str]]:
    """
    Generate ML-style suggestions for a list of violations.

    For Week 1, this is intentionally template-based.
    Later weeks can replace priority/confidence with a real model.
    """
    suggestions: List[Suggestion] = []
    notes: List[str] = []

    for idx, violation in enumerate(req.violations):
        key = (violation.article, violation.code)
        entry = ML_MAPPINGS.get(key)

        if entry:
            fix = entry["fix_template"]
            nec_refs = entry["nec_refs"]
            priority = float(entry["base_priority"])
            confidence = 0.6 + 0.1 * priority
            rationale = "Template-based remediation derived from rule context"
        else:
            fix = "Template fallback: review violation and apply standard NEC remediation"
            nec_refs = []
            priority = 0.3
            confidence = 0.4
            rationale = "Generic fallback (no specific mapping found)"

        suggestion = Suggestion(
            violation_ref=idx,
            fix=fix,
            nec_refs=nec_refs,
            confidence=confidence,
            priority=priority,
            rationale=rationale,
        )

        suggestions.append(suggestion)

    return suggestions, notes
