# LoanBuddy - Step 1: create 5,000 fake loan applicants
#
# Each person has:
#   1. The TRUTH: did they repay? (depends only on money)
#   2. What the BANK SEES: a credit score (unfairly low for young and gig workers)
#
# Run:  python data/generate.py

import random
import pandas as pd

random.seed(42)  # same fake people every time we run


people = []

for i in range(1, 5001):

    # ---------- 1. Who they are ----------
    age = random.randint(18, 60)
    gender = random.choice(["M", "F"])
    region = random.choice(["urban", "urban", "rural"])  # about 2 urban : 1 rural

    if age < 26:
        job = random.choice(["salaried", "gig", "gig", "self-employed"])
    else:
        job = random.choice(["salaried", "salaried", "salaried", "gig", "self-employed"])

    # ---------- 2. Their money ----------
    if job == "salaried":
        income = random.randint(30000, 60000)
    elif job == "gig":
        income = random.randint(18000, 40000)
    else:
        income = random.randint(20000, 60000)

    if region == "rural":
        income = income * 0.8

    existing_emi = income * random.uniform(0, 0.4)
    bank_balance = income * random.uniform(0.2, 2)

    loan_amount = income * random.uniform(1, 10)  # loan is 1x to 10x monthly income
    tenure = random.choice([6, 12, 24, 36, 48, 60])
    new_emi = loan_amount / tenure

    # burden = how much of their income goes to EMIs
    burden = (existing_emi + new_emi) / income

    if burden > 0.6:
        bounced = random.randint(0, 3)
    else:
        bounced = random.choice([0, 0, 0, 1])

    # ---------- 3. The TRUTH: did they repay? ----------
    # Only money decides this. No age, gender, region or job.
    chance = 1.0
    chance = chance - burden * 0.4       # heavy EMIs -> less likely to repay
    chance = chance - bounced * 0.08     # bounced payments -> less likely
    if bank_balance > income:
        chance = chance + 0.05           # good savings -> more likely

    if random.random() < chance:
        repaid = 1
    else:
        repaid = 0

    # ---------- 4. What the BANK SEES: credit score ----------
    score = 300 + chance * 600 + random.randint(-40, 40)

    # The planted BIAS: short credit history gets punished
    if age < 26:
        score = score - 80
    if job == "gig":
        score = score - 50

    score = max(300, min(900, score))  # keep it between 300 and 900

    # How many years they have been borrowing
    years_borrowing = random.randint(0, age - 18)
    if job == "gig":
        years_borrowing = years_borrowing // 2

    # No borrowing history -> no credit score at all
    if years_borrowing < 1:
        score = None

    # ---------- 5. Save this person ----------
    people.append({
        "applicant_id": f"APP_{i:04d}",
        "age": age,
        "age_group": "18-25" if age < 26 else "26+",
        "gender": gender,
        "region": region,
        "job": job,
        "income": round(income),
        "existing_emi": round(existing_emi),
        "bank_balance": round(bank_balance),
        "bounced": bounced,
        "loan_amount": round(loan_amount, -3),
        "tenure": tenure,
        "years_borrowing": years_borrowing,
        "credit_score": round(score) if score else None,
        "repaid": repaid,
    })


df = pd.DataFrame(people)
df.to_csv("data/applicants.csv", index=False)


# ---------- Self-test ----------
print("Saved", len(df), "people to data/applicants.csv")
print()
print("Overall repaid rate:", round(df["repaid"].mean() * 100, 1), "%   (target 75-85%)")
print("People with no score:", round(df["credit_score"].isna().mean() * 100, 1), "%   (target ~15%)")
print()

for column in ["age_group", "job", "gender", "region"]:
    print("--- by", column, "---")
    print(df.groupby(column)[["repaid", "credit_score"]].mean().round(2))
    print()