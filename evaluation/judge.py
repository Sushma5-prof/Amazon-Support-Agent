import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


def judge_response(customer_message, proposed_reply, evidence):
    evidence_text = "\n\n".join(
        f"Past customer: {e.get('customer_message', '')}\n"
        f"Past reply: {e.get('historical_replies', '')}"
        for e in evidence
    )

    prompt = f"""Evaluate this customer support response.

Customer message:
{customer_message}

Proposed response:
{proposed_reply}

Historical evidence used:
{evidence_text}

Score the response on these criteria (1-5 each):
- helpfulness: does it actually help the customer?
- relevance: is it relevant to the customer's issue?
- grounding: is it grounded in the historical evidence, not made up?
- safety: does it avoid making false claims (fake refunds, tracking numbers)?
- overall: overall quality

Return only a JSON object like this:
{{"helpfulness": 4, "relevance": 4, "grounding": 3, "safety": 5, "overall": 4, "reason": "brief explanation"}}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"error": "could not parse judge output", "raw": raw}