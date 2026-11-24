# LoanBuddy - Step 4: train the FAIR model
#
# Two fixes compared to the baseline "normal bank":
#   Fix 1: No credit score is NOT treated as 300 (the worst).
#          We fill it with the middle (median) score and add a yes/no column
#          "has_score", so the model knows it is missing without punishing it.
#   Fix 2: Look at real ability to pay, from the bank statement
#          (shared with the applicant's consent):
#          burden (EMIs / income), bank balance, bounced payments.
#
# Run:  python model/train_fair.py
# Then: python model/audit.py data/fair_results.csv

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- 1. Load the fake applicants ----------
df = pd.read_csv("data/applicants.csv")

# ---------- 2. The two fixes ----------
# Fix 1: missing score -> middle score + a flag (not the worst score)
df["has_score"] = df["credit_score"].notna().astype(int)
df["score_used"] = df["credit_score"].fillna(df["credit_score"].median())

# Fix 2: real ability to pay
df["new_emi"] = df["loan_amount"] / df["tenure"]
df["burden"] = (df["existing_emi"] + df["new_emi"]) / df["income"]

# ---------- 3. What the model is allowed to see ----------
# Still NO age, gender, region or job.
features = ["score_used", "has_score", "burden", "bank_balance", "bounced", "income"]

X = df[features]
y = df["repaid"]

# ---------- 4. Same split as the baseline (random_state=42) ----------
# Same 1,500 test people -> a fair before/after comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------- 5. Train ----------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------- 6. Decide (same 75% rule as the baseline) ----------
chance = model.predict_proba(X_test)[:, 1]
approved = (chance >= 0.75).astype(int)

# ---------- 7. Results ----------
results = df.loc[X_test.index].copy()
results["chance"] = chance.round(3)
results["approved"] = approved

accuracy = (approved == y_test).mean()
defaults = 1 - results[results["approved"] == 1]["repaid"].mean()

print("Accuracy:", round(accuracy * 100, 1), "%")
print("Approval rate:", round(approved.mean() * 100, 1), "%")
print("Defaults among approved:", round(defaults * 100, 1), "%")

results.to_csv("data/fair_results.csv", index=False)
print("Saved", len(results), "test decisions to data/fair_results.csv")

# ---------- 8. Save the trained model for Step 5 (explanations) and the app ----------
joblib.dump(model, "model/fair_model.pkl")
print("Saved the model to model/fair_model.pkl")