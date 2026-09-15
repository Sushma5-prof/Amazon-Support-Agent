# Decision Log

1. **Picked AmazonHelp specifically.** It has 169,840 support tweets — more than any other brand in the dataset. more data = better retrieval index and more training candidates. brands like SprintCare or AppleSupport had too few resolved threads to build a useful RAG store.

2. **Defined 11 intents manually from browsing the data.** didn't import an existing taxonomy like Banking77 because Amazon support issues are shipping/product-heavy, not finance-heavy. spent time reading 200+ random customer tweets first, then grouped them.

3. **Filtered on conversation_root, not tweet_id.** putting tweets from the same thread in both train and golden set would leak context. enforced disjoint roots throughout.

4. **50 training examples per intent, 20 golden per intent.** 50 is enough for TF-IDF + LR to learn basic vocabulary patterns per class. 20 golden per class gives enough signal to measure per-class recall without making the annotation task unmanageable.

5. **Used rule-based annotation for training labels, not manual.** 550 manual annotations in a few hours is unrealistic. the rule-based annotator uses 40+ regex patterns covering real tweet phrasing observed in the data. it's not perfect but the training labels don't need to be — the golden set does.

6. **Ensemble of word and char n-grams instead of a single model.** word n-grams capture intent vocabulary ("refund", "tracking"). char n-grams capture morphology and handle abbreviations, typos, and social media spelling like "pls" or "ur". the combination gained +5pp over word-only.

7. **Kept the TF-IDF approach rather than using a pretrained LLM classifier.** the goal is a working baseline you can run without a GPU. sentence-transformers would probably do better but takes longer, requires downloading 90MB, and makes the model less reproducible in 15 minutes.

8. **Chroma for retrieval, not just BM25.** BM25 retrieval would find exact keyword matches but miss paraphrases. customers describe the same problem in very different ways ("hasn't arrived", "still waiting", "where is it"). semantic search handles this better.

9. **Escalation before and after response generation.** two decision points: (a) if evidence is too thin, escalate before generating anything — avoids wasting an LLM call. (b) if the generated reply makes unsupported claims, escalate after. simpler systems skip (b).

10. **Grounding check is deterministic, not another LLM call.** using GPT to judge GPT's output would add latency and cost. a simple phrase blocklist catches the main failure mode (claiming to have refunded / accessed account) reliably enough for a first version.

11. **Reported 72.7% as the headline — not 92.3%.** the 92.3% was from evaluating on the same data used to derive training labels (heuristic-vs-heuristic). that number tells you nothing about real performance. 72.7% is measured on a held-out set with zero root overlap.

12. **Didn't build a conversation history feature.** the pipeline treats each tweet as a standalone message. in reality customers often reply multiple times in a thread. adding thread context would improve accuracy but adds complexity — left for next iteration.

13. **Response generator uses Groq's hosted LLM, not a local model or paid API.** local generation with small models was inconsistent and heavy on CPU, while commercial APIs like OpenAI incur costs. selected Groq for its free tier and fast inference, making the pipeline easily reproducible by reviewers with zero cost. users set GROQ_API_KEY.

14. **Golden set labels are documented as AI-annotated.** they were assigned by the same rule-based annotator used for training candidate selection, so they're not fully independent. the README and report both say this clearly. the correct fix is manual review — which would be the first thing to do with more time.

15. **Didn't include a FastAPI/web layer in the final submission.** app/main.py is a CLI demo. FastAPI was in the original README (leftover from early planning) but the backend integration is out of scope for a take-home eval. kept the file as a stub.