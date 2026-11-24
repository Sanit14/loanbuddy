# LoanBuddy - Step 2: train the BASELINE model (a "normal bank")
#
# This model works like most banks today:
#   - it trusts the credit score a lot
#   - if you have NO credit score, it treats you as the worst score (300)
#
# Run:  python model/train_baseline.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- 1. Load the fake applicants ----------
df = pd.read_csv("data/applicants.csv")

# No credit score? The normal bank treats it as the lowest score.
df["score_used"] = df["credit_score"].fillna(300)

# ---------- 2. Choose what the model is allowed to see ----------
# Only money + credit score. NOT age, gender, region or job.
features = ["score_used", "income", "existing_emi", "loan_amount", "tenure"]

X = df[features]     # the inputs
y = df["repaid"]     # the answer we want the model to learn

# ---------- 3. Split: 70% to learn from, 30% to test on ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------- 4. Train the model ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- 5. Make decisions on the test people ----------
# The model gives a "chance of repaying" for each person.
# Bank rule: approve only if chance is 75% or more.
chance = model.predict_proba(X_test)[:, 1]
approved = (chance >= 0.75).astype(int)

# ---------- 6. Check how good it is ----------
accuracy = (approved == y_test).mean()
print("Accuracy:", round(accuracy * 100, 1), "%")
print("Approval rate:", round(approved.mean() * 100, 1), "%")

# ---------- 7. Save the decisions for Step 3 (fairness audit) ----------
results = df.loc[X_test.index].copy()
results["chance"] = chance.round(3)
results["approved"] = approved
results.to_csv("data/baseline_results.csv", index=False)

print("Saved", len(results), "test decisions to data/baseline_results.csv")