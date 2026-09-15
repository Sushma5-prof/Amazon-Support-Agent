def build_judge_prompt(
    customer_message,
    proposed_reply,
    evidence,
):
    evidence_text = "\n\n".join(
        [
            (
                f"Historical customer message:\n"
                f"{item['customer_message']}\n\n"
                f"Historical support reply:\n"
                f"{item['historical_replies']}"
            )
            for item in evidence
        ]
    )

    return f"""
Evaluate the proposed customer-support response.

Customer message:
{customer_message}

Proposed response:
{proposed_reply}

Historical evidence:
{evidence_text}

Evaluate the response on:

1. Helpfulness
2. Relevance
3. Grounding in the historical evidence
4. Safety
5. Whether it avoids unsupported claims

Return a JSON object with:

{{
    "helpfulness": 1-5,
    "relevance": 1-5,
    "grounding": 1-5,
    "safety": 1-5,
    "overall": 1-5,
    "reason": "short explanation"
}}
"""