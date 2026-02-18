# Payment Transaction Fraud Detection & Financial Monitoring

---

## Business Impact — Why I Built This

Fraud is rare (0.52% of transactions), but expensive when missed.  
At the same time, falsely blocking legitimate customers damages trust and increases operational cost.

When I evaluated the baseline rules engine, I found:

- It caught **54.6% of fraud**
- It flagged **22.3% of all transactions**
- It generated ~2,230 alerts per 10,000 transactions

That means:
- Nearly half of fraud went undetected.
- Roughly 1 in 5 customers risked unnecessary friction.
- Review teams were overloaded with low-quality alerts.

My objective was simple:

> Catch more fraud while reducing unnecessary customer interruptions.

After redesigning the system using behavioral anomaly detection:

- Fraud detection increased to **83.1%**
- False positives dropped by **35% (relative reduction)**
- Alerts fell to ~1,450 per 10,000 transactions

This improved signal quality without increasing system complexity.

---

## My Approach

I followed three core principles.

---

### 1. Clarity Over Complexity

I avoided black-box feature engineering.

Instead of abstract components or opaque embeddings, I used interpretable behavioral signals:

- Time of transaction (night, weekend)
- Transaction velocity (1-hour and 24-hour windows)
- 30-day and 90-day personal spending baselines
- Distance from home
- Merchant category historical fraud rate

If a transaction is flagged, I can explain it clearly:

> “This purchase was 847 km from the customer’s home, 4 standard deviations above their normal spending, and in a historically higher-risk category.”

If I cannot explain it in plain language, I do not consider it production-ready.

---

### 2. Respecting Degrees of Freedom

Fraud detection has limited true signal:

- Fraud is rare.
- Customer behavior is noisy.
- Many features are correlated.

Adding excessive features increases variance without improving generalization.

So I constrained the system:

- Used rolling behavioral windows (7d, 30d, 90d)
- Split data chronologically to prevent leakage
- Avoided artificial resampling
- Tuned thresholds based on operational trade-offs

I treated complexity as a cost, not an asset.

---

### 3. Function Over Fancy Models

I first built a realistic rules-engine baseline:

- High transaction amount
- Night-time transaction
- High-risk merchant category
- High transaction velocity
- Flag if 2 or more rules trigger

Then I improved on it using:

- Z-score deviation from personal spending history
- Isolation Forest for multi-dimensional anomaly detection
- Simple combined alert logic

No deep learning.  
No stacked ensembles.  
Just measurable improvement over a practical baseline.

---

## Measured Results

| Metric | Rules Engine | My Anomaly System |
|--------|--------------|-------------------|
| Fraud Detection (Recall) | 54.6% | 83.1% |
| False Positive Rate | 22.3% | 14.5% |
| Precision | 1.9% | 4.1% |
| Alerts / 10K Transactions | 2,230 | 1,450 |

### Business Interpretation

- +28.5 percentage points increase in fraud detection
- 35% relative reduction in false positives
- Fewer unnecessary customer disruptions
- Lower operational review burden
- Improved alert precision without added architectural complexity

---

## Why It Works

Fraud is not just about large transactions.

It is about behavioral deviation across multiple dimensions:

- Is this amount normal for this account?
- Is this location plausible?
- Is this merchant category statistically riskier?
- Is the transaction velocity abnormal?

The system evaluates abnormality relative to the individual account — not just global thresholds.

That shift in framing materially improved performance.

---

## Explainability

Because features are directly tied to real fields (amount, distance, velocity, category risk), alerts are interpretable.

Example flagged transaction:

- 847 km from home
- 4.1 standard deviations above 30-day mean
- Merchant category with 1.8% historical fraud rate

This makes the system usable by:
- Risk analysts
- Operations teams
- Compliance stakeholders
- Leadership

Explainability was a design constraint — not an afterthought.

---

## Limitations

- The dataset is synthetic (Sparkov-generated).
- Fraud patterns do not evolve (no concept drift).
- The projected chargeback reduction is modeled from detection improvement, not directly observed.
- A real deployment would require ongoing retraining and drift monitoring.

The architecture supports those requirements, but this evaluation was conducted on a static dataset.

---

## Final Reflection

This project demonstrates that:

- Behavioral anomaly detection can significantly outperform static rules.
- Reducing false positives is as important as increasing recall.
- Interpretable features outperform unnecessary complexity.
- Clear, scalable design creates measurable business value.

In short:

> I replaced a high-noise rules engine with a behavior-aware anomaly system that catches more fraud and interrupts fewer legitimate customers — using explainable and scalable design.
