# LoanBuddy - Step 5: explain each decision in plain words (SHAP)
#
# SHAP splits a decision into pieces:
#   how much did each input push the chance UP (+) or DOWN (-)?
#
# Run:  python model/explain.py APP_0007
#       (any applicant ID from data/fair_results.csv)

import sys
import joblib
import pandas as pd
import shap

features = ["score_used", "has_score", "burden", "bank_balance", "bounced", "income"]

# Friendly names for the applicant to read
names = {
    "score_used": "Credit score",
    "has_score": "Credit history",
    "burden": "EMIs compared to income",
    "bank_balance": "Bank balance",
    "bounced": "Bounced payments",
    "income": "Monthly income",
}

# ---------- 1. Load the saved model and the test applicants ----------
model = joblib.load("model/fair_model.pkl")
df = pd.read_csv("data/fair_results.csv")

# ---------- 2. Pick one applicant ----------
app_id = sys.argv[1] if len(sys.argv) > 1 else df["applicant_id"].iloc[0]
person = df[df["applicant_id"] == app_id]

if person.empty:
    print("No applicant", app_id, "in the test set. Try one from data/fair_results.csv")
    sys.exit()

# ---------- 3. Ask SHAP: what pushed this decision up or down? ----------
background = df[features].sample(100, random_state=42)  # 100 "typical" people to compare against
explainer = shap.LinearExplainer(model, background)
pushes = explainer.shap_values(person[features])[0]

# ---------- 4. Print the decision ----------
row = person.iloc[0]
decision = "APPROVED" if row["approved"] == 1 else "NOT APPROVED"

print("Applicant:", app_id)
print("Decision: ", decision)
print("Chance of repaying:", round(row["chance"] * 100), "%  (needs 75%)")
print()
print("Main reasons:")

# ---------- 5. Top 3 reasons, biggest first ----------
reasons = sorted(zip(features, pushes), key=lambda x: abs(x[1]), reverse=True)

for feature, push in reasons[:3]:
    value = row[feature]
    if feature == "burden":
        shown = str(round(value * 100)) + "% of income"
    elif feature == "has_score":
        shown = "yes" if value == 1 else "no credit history yet"
    elif feature in ["bank_balance", "income"]:
        shown = "Rs " + format(int(value), ",")
    else:
        shown = str(int(value))

    effect = "helped (+)" if push > 0 else "hurt (-)"
    print(" ", effect.ljust(11), names[feature], "=", shown)