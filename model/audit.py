# LoanBuddy - Step 3: fairness audit
#
# Question: does the model treat every group fairly?
# Test used: the 4/5ths (80%) rule
#   For each group, divide its approval rate by the best group's rate.
#   If the answer is below 0.80 -> possible bias.
#
# Run:  python model/audit.py
# (Later you can audit any results file: python model/audit.py data/fair_results.csv)

import sys
import pandas as pd

file = sys.argv[1] if len(sys.argv) > 1 else "data/baseline_results.csv"
df = pd.read_csv(file)

print("Auditing:", file)
print("Overall approval rate:", round(df["approved"].mean() * 100, 1), "%")
print()

# These columns were NOT used by the model. We use them ONLY to check fairness.
for column in ["age_group", "job", "gender", "region"]:

    print("=== Fairness by", column, "===")

    table = df.groupby(column).agg(
        people=("approved", "size"),
        actually_repaid=("repaid", "mean"),
        approved=("approved", "mean"),
    )

    best_rate = table["approved"].max()
    table["ratio_vs_best"] = table["approved"] / best_rate
    table["4/5ths test"] = table["ratio_vs_best"].apply(
        lambda r: "PASS" if r >= 0.8 else "FAIL"
    )

    print(table.round(2))
    print()