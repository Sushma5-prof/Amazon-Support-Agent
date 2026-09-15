import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class ResponseGenerator:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def _build_prompt(self, message, intent, evidence):
        bits = []
        for i, e in enumerate(evidence, 1):
            bits.append(f"Example {i}:\nCustomer: {e.get('customer_message', '')}\nReply: {e.get('historical_replies', '')}")
        evidence_block = "\n\n".join(bits) if bits else "no examples found"

        return f"""You are a support agent for Amazon. A customer sent this:

"{message}"

Their issue is about: {intent}

Here are some similar past cases and how they were handled:
{evidence_block}

Write a short, helpful reply to the customer. Don't mention the examples above. Don't make up order numbers or tracking info. If you don't know something, say you'll look into it."""

    def generate(self, customer_message, intent, evidence):
        prompt = self._build_prompt(customer_message, intent, evidence)
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message["content"].strip()

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
                return {"passed": False, "reason": f"reply claims something we can't verify: '{phrase}'"}
        
        return {"passed": True, "reason": "ok"}
