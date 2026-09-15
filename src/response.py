import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


class ResponseGenerator:
    def generate(self, customer_message, intent, evidence):
        examples = []
        for i, e in enumerate(evidence, 1):
            examples.append(
                f"Example {i}:\nCustomer: {e.get('customer_message', '')}\n"
                f"Reply: {e.get('historical_replies', '')}"
            )
        evidence_block = "\n\n".join(examples) if examples else "no examples found"

        prompt = f"""You are a support agent for Amazon. A customer sent this:

"{customer_message}"

Their issue is about: {intent}

Here are some similar past cases and how they were handled:
{evidence_block}

Write a short, helpful reply to the customer. Don't mention the examples above. Don't make up order numbers or tracking info. If you don't know something, say you'll look into it."""

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def grounding_check(self, reply, customer_message, evidence):
        bad_phrases = [
            "i have refunded",
            "your refund has been processed",
            "your order has been cancelled",
            "i accessed your account",
            "your tracking number is",
            "your package will arrive on",
        ]
        if not reply.strip():
            return {"passed": False, "reason": "empty reply"}

        reply_lower = reply.lower()
        for phrase in bad_phrases:
            if phrase in reply_lower:
                return {"passed": False, "reason": f"unsupported claim: '{phrase}'"}

        return {"passed": True, "reason": "ok"}
