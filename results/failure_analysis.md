# Failure Analysis

from the golden set evaluation (72.7% accuracy, 60 failures out of 220).

---

## 1. DELIVERY_DELAY confused with SHIPPED_TRACKING

**Count:** 9 failures

**Example:**
> "Last 2 @AmazonHelp prime packages missed delivery date, 1 is 3 days late & counting"

**Predicted:** SHIPPING_TRACKING  
**Correct:** DELIVERY_DELAY

**Why:** the word "missed delivery date" overlaps with tracking vocabulary ("delivery date"), and the model latches onto the date/shipping framing rather than the complaint framing. both classes share words like "delivery", "package", "arrive".

**Hypothesis:** the classifier needs more examples where the user is *complaining about lateness* vs *asking where the package is*. the distinction is tone, not just vocabulary — hard for TF-IDF to capture.

---

## 2. DELIVERY_DELAY confused with DELIVERED_NOT_RECEIVED

**Count:** 7 failures

**Example:**
> "@AmazonHelp hello please help me my package was suppose to be here by 8pm PST and i had got an email saying it got delivered"

**Predicted:** DELIVERED_NOT_RECEIVED  
**Correct:** DELIVERY_DELAY

**Why:** the email saying "delivered" causes the model to predict DELIVERED_NOT_RECEIVED even though the customer is reporting a delay. ambiguous messages that contain signals from both classes consistently fall on the wrong side.

**Hypothesis:** these are genuinely ambiguous. a human might also disagree on the correct label. the right fix might be merging the two classes or adding a confidence threshold that triggers escalation for borderline cases.

---

## 3. OTHER class has near-zero recall (0.20)

**Count:** 16 out of 20 OTHER examples misclassified

**Example:**
> "@AmazonHelp thanks for your help yesterday, really appreciated it"

**Predicted:** GENERAL_SUPPORT  
**Correct:** OTHER

**Why:** the OTHER class is defined by absence — it catches greetings, thanks, and unrelated tweets. but the classifier has no positive signal for it, just negative space. GENERAL_SUPPORT is the nearest bucket and absorbs most of them.

**Hypothesis:** OTHER is too broad. splitting it into FEEDBACK_POSITIVE and UNRELATED would help, or just merging it into GENERAL_SUPPORT and accepting that distinction isn't worth making.

---

## 4. PRIME confused with DELIVERY_DELAY when Prime shipping is mentioned

**Count:** 5 failures

**Example:**
> "@AmazonHelp it is giving me a hard time with my prime benefits im not getting my package on time"

**Predicted:** DELIVERY_DELAY  
**Correct:** PRIME

**Why:** the customer is complaining that Prime benefits aren't working (delivery speed), but the surface text "not getting my package on time" triggers DELIVERY_DELAY. the primary complaint is about Prime subscription value, not the specific delivery.

**Hypothesis:** intent here depends on *what the customer wants fixed* — their Prime account vs a specific delivery. this requires understanding the complaint at a semantic level that bag-of-words can't do well.

---

## 5. SHIPPING_TRACKING confused with DELIVERY_DELAY

**Count:** 6 failures

**Example:**
> "@AmazonHelp I am sure you have escalated the issue and will make the delivery even faster."

**Predicted:** SHIPPING_TRACKING  
**Correct:** DELIVERY_DELAY

**Why:** sarcastic/indirect phrasing. the customer isn't asking for tracking, they're sarcastically complaining about a delay. the model sees "delivery" and no explicit delay keywords and picks the wrong class.

**Hypothesis:** sarcasm and indirect complaint phrasing is a fundamental weakness of n-gram classifiers. would need sentiment-aware features or an LLM-based classifier to handle these reliably.