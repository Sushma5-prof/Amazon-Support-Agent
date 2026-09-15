# Decision Log

## 1. Brand selection

Selected AmazonHelp because it has the largest useful support volume in
the downloaded dataset among the investigated brands.

## 2. Evaluation unit

Use the individual customer message as the primary intent-classification
unit rather than the entire conversation.

## 3. Conversation leakage

Conversation roots represented in the golden set are excluded from the
development pool.

## 4. Intent taxonomy

Use 11 manually defined intents based on recurring customer-support
problems in the AmazonHelp data.

## 5. Candidate labels

Heuristic intent labels are used only for sampling and exploration.
They are not treated as ground truth.

## 6. Golden set

Use a held-out hand-labelled golden set for final evaluation.

## 7. Historical replies

Historical AmazonHelp responses are treated as evidence rather than
scripts.

## 8. Retrieval

Use semantic retrieval to find historically similar customer-support
interactions.

## 9. Response generation

Use a local Ollama model to avoid dependence on paid APIs.

## 10. Escalation

Escalate when evidence is insufficient, account-specific investigation
is required, or the generated response fails the grounding/safety check.

## 11. Agent architecture

Use a conditional LangGraph pipeline rather than a multi-agent system
or iterative agent loop.

## 12. API design

Keep the FastAPI layer thin and place business logic inside the pipeline.

## 13. Evaluation

Compare the final intent classifier against majority and TF-IDF baselines.

## 14. Safe auto-handle rate

Measure the proportion of messages automatically handled correctly
rather than reporting only overall accuracy.

## 15. Historical evidence limitations

Historical support conversations may be incomplete, outdated,
inconsistent, or specific to an individual customer. The system must
not treat historical responses as authoritative instructions.