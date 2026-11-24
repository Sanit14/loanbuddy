# LoanBuddy - Step 6: the app
#
# Four pages:
#   1. Apply for a loan   -> consent, decision, reasons, Key Fact Statement
#   2. Loan officer       -> secure login, review queue (human in the loop)
#   3. Fairness dashboard -> before vs after audit
#   4. My data & privacy  -> what we collect, your rights, delete your data
#
# Run:  streamlit run app/app.py

import hashlib
import os
import random
from datetime import date

import joblib
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="LoanBuddy", page_icon="🤝")

features = ["score_used", "has_score", "burden", "bank_balance", "bounced", "income"]
LOG_FILE = "data/app_log.csv"
OFFICER_FILE = "data/officer_decisions.csv"
ACCESS_FILE = "data/access_log.csv"

# ---------- Officer login: we store only a HASH of the password, never the password ----------
# A hash is a one-way scramble: you can check a password against it, but can't turn it back.
# Demo login -> username: officer   password: loanbuddy123
SALT = b"loanbuddy-demo-salt"
OFFICERS = {
    "officer": hashlib.pbkdf2_hmac("sha256", b"loanbuddy123", SALT, 100000).hex(),
}


def password_ok(username, password):
    if username not in OFFICERS:
        return False
    attempt = hashlib.pbkdf2_hmac("sha256", password.encode(), SALT, 100000).hex()
    return attempt == OFFICERS[username]


def write_access_log(who, action):
    # Audit log: who did what, and when
    row = pd.DataFrame([{"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "who": who, "action": action}])
    row.to_csv(ACCESS_FILE, mode="a", index=False, header=not os.path.exists(ACCESS_FILE))

# ---------- Load the model (only once, then Streamlit remembers it) ----------
@st.cache_resource
def load_everything():
    model = joblib.load("model/fair_model.pkl")
    applicants = pd.read_csv("data/applicants.csv")
    median_score = applicants["credit_score"].median()
    test_people = pd.read_csv("data/fair_results.csv")
    background = test_people[features].sample(100, random_state=42)
    explainer = shap.LinearExplainer(model, background)
    return model, median_score, explainer

model, median_score, explainer = load_everything()


# ---------- Sidebar menu ----------
page = st.sidebar.radio("Menu", ["Apply for a loan", "Loan officer", "Fairness dashboard", "My data & privacy"])
st.sidebar.markdown("---")
st.sidebar.caption("LoanBuddy is a student project. All data is synthetic (fake).")


# =====================================================================
# PAGE 1: APPLY
# =====================================================================
if page == "Apply for a loan":

    st.title("🤝 LoanBuddy")
    st.write("A fair, explainable personal loan check. "
             "We never ask for your age, gender, religion, caste, contacts or photos.")

    st.subheader("Your details")
    income = st.number_input("Monthly income (Rs)", 5000, 500000, 30000, step=1000)
    existing_emi = st.number_input("EMIs you already pay each month (Rs)", 0, 500000, 3000, step=500)
    bank_balance = st.number_input("Average bank balance (Rs)", 0, 5000000, 25000, step=1000)
    bounced = st.number_input("Bounced payments in the last 6 months", 0, 20, 0)

    has_score = st.radio("Do you have a credit score?", ["Yes", "No, I'm new to credit"])
    credit_score = None
    if has_score == "Yes":
        credit_score = st.slider("Your credit score", 300, 900, 700)

    st.subheader("Your loan")
    loan_amount = st.number_input("Loan amount (Rs)", 10000, 500000, 100000, step=5000)
    tenure = st.selectbox("Repay over (months)", [6, 12, 24, 36, 48, 60], index=2)

    # ---------- Consent: unticked by default, and specific ----------
    st.subheader("Your consent")
    consent_decide = st.checkbox("Use these details only to decide this loan (required)")
    consent_keep = st.checkbox("Keep my application for 90 days so a loan officer can review it (optional)")

    if st.button("Check my loan", type="primary"):

        if not consent_decide:
            st.error("We can't check your loan without your consent. Please tick the first box.")
            st.stop()

        # ---------- Same preparation as in training ----------
        new_emi = loan_amount / tenure
        person = pd.DataFrame([{
            "score_used": credit_score if credit_score else median_score,
            "has_score": 1 if credit_score else 0,
            "burden": round((existing_emi + new_emi) / income, 3),
            "bank_balance": bank_balance,
            "bounced": bounced,
            "income": income,
        }])

        chance = model.predict_proba(person[features])[0][1]
        burden = person["burden"][0]

        # ---------- Decision with a human in the loop ----------
        # Safety rule: EMIs above 60% of income always go to a person, even if the model says yes
        if chance >= 0.75 and burden > 0.6:
            decision = "OFFICER REVIEW"
            st.warning(f"👤 The model says {chance:.0%}, but your EMIs would be {burden:.0%} of your income. "
                       "For your safety, a loan officer will review this before any loan is given.")
        elif chance >= 0.75:
            decision = "APPROVED"
            st.success(f"✅ Approved! Chance of repaying: {chance:.0%}")
        elif chance >= 0.60:
            decision = "OFFICER REVIEW"
            st.warning(f"👤 Sent to a loan officer for review. Chance of repaying: {chance:.0%}. "
                       "A person, not the computer, will make the final decision.")
        else:
            decision = "NOT APPROVED"
            st.error(f"❌ Not approved right now. Chance of repaying: {chance:.0%} (we need 75%).")

        # ---------- Explanation (SHAP) ----------
        st.subheader("Why?")
        names = {
            "score_used": "Credit score",
            "has_score": "Credit history",
            "burden": "EMIs compared to income",
            "bank_balance": "Bank balance",
            "bounced": "Bounced payments",
            "income": "Monthly income",
        }
        pushes = explainer.shap_values(person[features])[0]
        reasons = sorted(zip(features, pushes), key=lambda x: abs(x[1]), reverse=True)

        for feature, push in reasons[:3]:
            icon = "🟢 helped" if push > 0 else "🔴 hurt"
            st.write(f"{icon} — **{names[feature]}**")

        # ---------- What could change the result (counterfactual) ----------
        if decision != "APPROVED":
            st.subheader("What could change the result?")
            if burden > 0.5:
                st.write(f"- Your EMIs would be **{burden:.0%}** of your income. "
                         "A smaller loan or a longer repayment period would bring this under 50%.")
            if bounced > 0:
                st.write("- A few months with no bounced payments would help.")
            if not credit_score:
                st.write("- Building a credit history (for example a small credit card paid on time) would help.")

        # ---------- Key Fact Statement (only if approved) ----------
        if decision == "APPROVED":
            st.subheader("Key Fact Statement")
            yearly_rate = 14                     # example interest rate for this demo
            r = yearly_rate / 12 / 100
            emi = loan_amount * r * (1 + r) ** tenure / ((1 + r) ** tenure - 1)
            fee = loan_amount * 0.02
            st.table(pd.DataFrame({
                "Item": ["Loan amount", "Interest rate", "Monthly EMI", "Months",
                         "Processing fee (2%)", "Total you will pay"],
                "Value": [f"Rs {loan_amount:,.0f}", f"{yearly_rate}% per year", f"Rs {emi:,.0f}",
                          str(tenure), f"Rs {fee:,.0f}", f"Rs {emi * tenure + fee:,.0f}"],
            }))

        # ---------- Save a log entry (pseudonymised: an ID, no name) ----------
        if consent_keep:
            app_id = f"LB_{random.randint(10000, 99999)}"
            entry = person.copy()
            entry.insert(0, "application_id", app_id)
            entry["chance"] = round(chance, 3)
            entry["decision"] = decision
            entry["date"] = date.today().isoformat()
            entry.to_csv(LOG_FILE, mode="a", index=False, header=not os.path.exists(LOG_FILE))
            st.info(f"Your application ID is **{app_id}**. Keep it to view or delete your data.")
        elif decision == "OFFICER REVIEW":
            st.info("A loan officer can only review applications that are kept. "
                    "To ask for a review, tick the second consent box and check again.")
        else:
            st.info("You chose not to save your application, so nothing was stored.")


# =====================================================================
# PAGE 2: LOAN OFFICER (human in the loop)
# =====================================================================
elif page == "Loan officer":

    st.title("👤 Loan officer")

    # ---------- Login ----------
    if "officer" not in st.session_state:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log in"):
            if password_ok(username, password):
                st.session_state["officer"] = username
                write_access_log(username, "logged in")
                st.rerun()
            else:
                write_access_log(username or "unknown", "FAILED login")
                st.error("Wrong username or password.")
        st.stop()

    officer = st.session_state["officer"]
    st.write(f"Logged in as **{officer}**")
    if st.button("Log out"):
        write_access_log(officer, "logged out")
        del st.session_state["officer"]
        st.rerun()

    # ---------- Review queue ----------
    st.subheader("Applications waiting for review")

    if not os.path.exists(LOG_FILE):
        st.write("No applications yet.")
        st.stop()

    log = pd.read_csv(LOG_FILE)
    waiting = log[log["decision"] == "OFFICER REVIEW"]

    if os.path.exists(OFFICER_FILE):
        done = pd.read_csv(OFFICER_FILE)["application_id"]
        waiting = waiting[~waiting["application_id"].isin(done)]

    if waiting.empty:
        st.write("Nothing waiting. 🎉")
        st.stop()

    # Least privilege: the officer sees only what is needed to decide. No names exist at all.
    for _, app in waiting.iterrows():
        with st.expander(f"{app['application_id']}  —  model chance {app['chance']:.0%}"):
            st.write(f"EMIs compared to income: **{app['burden']:.0%}**")
            st.write(f"Bank balance: Rs {app['bank_balance']:,.0f}  |  Bounced payments: {app['bounced']}")
            st.write("Credit history: " + ("yes" if app["has_score"] == 1 else "no credit history yet"))

            reason = st.text_input("Reason for your decision (required)", key="r_" + app["application_id"])
            col1, col2 = st.columns(2)
            approve = col1.button("Approve", key="a_" + app["application_id"])
            reject = col2.button("Reject", key="x_" + app["application_id"])

            if approve or reject:
                if not reason.strip():
                    st.error("Please write a reason. Every human decision must be explained too.")
                else:
                    row = pd.DataFrame([{
                        "application_id": app["application_id"],
                        "model_chance": app["chance"],
                        "officer": officer,
                        "officer_decision": "APPROVED" if approve else "REJECTED",
                        "reason": reason,
                        "date": date.today().isoformat(),
                    }])
                    row.to_csv(OFFICER_FILE, mode="a", index=False, header=not os.path.exists(OFFICER_FILE))
                    write_access_log(officer, f"decided {app['application_id']}")
                    st.rerun()


# =====================================================================
# PAGE 3: FAIRNESS DASHBOARD
# =====================================================================
elif page == "Fairness dashboard":

    st.title("⚖️ Fairness dashboard")
    st.write("We test LoanBuddy with the **4/5ths rule**: every group's approval rate "
             "must be at least 80% of the best group's rate.")

    before = pd.read_csv("data/baseline_results.csv")
    after = pd.read_csv("data/fair_results.csv")

    group = st.selectbox("Check fairness by", ["age_group", "job", "gender", "region"])

    comparison = pd.DataFrame({
        "Normal bank": before.groupby(group)["approved"].mean(),
        "LoanBuddy": after.groupby(group)["approved"].mean(),
    })
    st.bar_chart(comparison)

    ratio_before = comparison["Normal bank"].min() / comparison["Normal bank"].max()
    ratio_after = comparison["LoanBuddy"].min() / comparison["LoanBuddy"].max()

    col1, col2 = st.columns(2)
    col1.metric("Normal bank ratio", f"{ratio_before:.2f} " + ("✅ PASS" if ratio_before >= 0.8 else "❌ FAIL"))
    col2.metric("LoanBuddy ratio", f"{ratio_after:.2f} " + ("✅ PASS" if ratio_after >= 0.8 else "❌ FAIL"))

    st.caption("Age, gender, region and job are NEVER used to make decisions. "
               "We use them only here, to check fairness.")


# =====================================================================
# PAGE 4: MY DATA & PRIVACY
# =====================================================================
else:

    st.title("🔒 My data & privacy")

    st.subheader("What we collect")
    st.write("Income, existing EMIs, bank balance, bounced payments, credit score (if you have one), "
             "loan amount and tenure. Nothing else.")

    st.subheader("What we never collect")
    st.write("Name, phone contacts, photos, call logs, age, gender, religion or caste.")

    st.subheader("How long we keep it")
    st.write("Only if you allow it, for 90 days, and only to review your application.")

    st.subheader("Your rights (DPDP Act 2023)")
    st.write("You can see your data, delete it, or withdraw consent at any time.")

    app_id = st.text_input("Enter your application ID (for example LB_12345)")

    if app_id and os.path.exists(LOG_FILE):
        log = pd.read_csv(LOG_FILE)
        mine = log[log["application_id"] == app_id]

        if mine.empty:
            st.write("No data found for this ID.")
        else:
            st.write("This is everything we hold about you:")
            st.dataframe(mine)
            if st.button("Delete my data", type="primary"):
                log = log[log["application_id"] != app_id]
                log.to_csv(LOG_FILE, index=False)
                st.success("Your data has been deleted.")
    elif app_id:
        st.write("No data found for this ID.")

    st.caption("Questions or complaints? Contact our grievance officer: grievance@loanbuddy.example")