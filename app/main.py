"""
Interactive CLI demo for the Hiver AI Support Agent.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import SupportAgent

def main():
    print("Hiver AI Support Agent (type \'quit\' to exit)")
    agent = SupportAgent()
    while True:
        msg = input("\nCustomer message: ").strip()
        if msg.lower() in ("quit", "exit"):
            break
        if not msg:
            continue
        result = agent.graph.invoke({"message": msg})
        print(f"  Intent    : {result.get('intent')}")
        print(f"  Decision  : {result.get('decision')}")
        print(f"  Reason    : {result.get('reason')}")
        reply = result.get("reply", "")
        if reply:
            print(f"  Reply     : {reply}")

if __name__ == "__main__":
    main()
