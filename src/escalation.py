ESCALATION_INTENTS = {
    "ACCOUNT_ACCESS",
}


def check_evidence_sufficiency(
    evidence,
    minimum_similarity=0.45,
):
    if not evidence:
        return {
            "sufficient": False,
            "reason": "No relevant historical evidence found",
        }

    best_similarity = max(
        item["similarity"]
        for item in evidence
    )

    if best_similarity < minimum_similarity:
        return {
            "sufficient": False,
            "reason": "Historical evidence is insufficient",
        }

    return {
        "sufficient": True,
        "reason": "Relevant historical evidence found",
    }


def should_escalate_before_response(
    intent,
    evidence_check,
):
    if not evidence_check["sufficient"]:
        return {
            "decision": "ESCALATE",
            "reason": evidence_check["reason"],
        }

    if intent in ESCALATION_INTENTS:
        return {
            "decision": "ESCALATE",
            "reason": "Requires account-specific investigation",
        }

    return {
        "decision": "CONTINUE",
        "reason": "No pre-response escalation required",
    }


def should_escalate_after_response(
    grounding_check,
):
    if not grounding_check["passed"]:
        return {
            "decision": "ESCALATE",
            "reason": grounding_check["reason"],
        }

    return {
        "decision": "AUTO",
        "reason": "Response passed grounding and safety checks",
    }