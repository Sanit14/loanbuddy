LoanBuddy 🤝

A fair, explainable personal loan app — Ethical AI mini project BCA (Honours), MIT ADT University · Ethical AI, Responsible Computing and Data Privacy

Live demo: add your Streamlit link here

The problem

Loan apps decide using credit scores. But people with little or no credit history — young people, gig workers — get low scores or none at all, even when they can repay. A model that trusts credit scores rejects them unfairly, even if it never sees their age, because the credit score acts as a proxy for age.

What LoanBuddy shows
Measure	Normal bank (baseline)	LoanBuddy
Approval rate, age 18–25	51%	75%
Approval rate, age 26+	68%	78%
4/5ths fairness test (age)	0.75 FAIL	0.97 PASS
Accuracy	70.6%	79.0%
Defaults among approved	13.8%	14.1%

LoanBuddy is fairer and more accurate, with almost no extra risk.

How it maps to the syllabus
Unit	Where it appears
1. Ethics & AI	Accountable lender, human-centred screens
2. Bias & fairness	Proxy bias found, 4/5ths audit, fix, SHAP explanations, human-in-the-loop
3. Data privacy	Data minimisation, no sensitive fields, consent, pseudonymised IDs
4. Privacy laws	DPDP rights page (view/delete), Key Fact Statement, grievance contact
5. Responsible systems	Synthetic data, hashed passwords, roles, audit logs, retention limit
Project structure
data/generate.py          Step 1  create 5,000 synthetic applicants (with planted thin-file bias)
model/train_baseline.py   Step 2  "normal bank" model (missing score = 300)
model/audit.py            Step 3  fairness audit using the 4/5ths rule
model/train_fair.py       Step 4  fair model (no penalty for missing score + ability-to-pay data)
model/explain.py          Step 5  SHAP explanation for one applicant
app/app.py                Step 6–7  Streamlit app: apply, loan officer, fairness dashboard, my data
Run it yourself
python -m venv .venv
.venv\Scripts\Activate.ps1          (Windows)
pip install -r requirements.txt fairlearn
python data/generate.py
python model/train_baseline.py
python model/audit.py
python model/train_fair.py
python model/audit.py data/fair_results.csv
python model/explain.py APP_2452
streamlit run app/app.py

Loan officer demo login: officer / loanbuddy123

Limitations
All data is synthetic. Real credit data would behave differently.
In real life, bank-statement data is available only for applicants who consent to share it.
The demo password is in the code (as a hash). A real system would keep secrets outside the code and add MFA.
On the free cloud host, saved applications and logs reset when the app restarts.
Interest rate (14%) and fee (2%) are example values.