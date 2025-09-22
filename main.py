# main.py
# Finance & Investment Advisor Bot - Streamlit app
# Run: streamlit run main.py

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, date

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sqlmodel import SQLModel, Field, Session, create_engine, select

# ---------------------------
# Load environment (.env) if present
# ---------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not installed or .env not present -> continue, we'll fallback to env vars
    pass

# ---------------------------
# Optional: yfinance for live prices
# ---------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# ---------------------------
# GROQ API Key: safe loading
# Try Streamlit secrets -> fallback to environment variable (.env or system env)
# ---------------------------
def load_groq_api_key() -> Optional[str]:
    # Try Streamlit secrets first (safe). Access inside try/except because if secrets.toml
    # doesn't exist, Streamlit will raise an error when parsing.
    try:
        # st.secrets behaves like a dict; using get might still trigger parsing of secrets file,
        # so use try/except around direct access.
        if "GROQ_API_KEY" in st.secrets:
            val = st.secrets["GROQ_API_KEY"]
            if val:
                return val
    except Exception:
        # secrets file not present or parsing failed: fallback to env
        pass

    # Fallback to environment variables (.env loaded earlier if present).
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = load_groq_api_key()
USE_GROQ = bool(GROQ_API_KEY)
groq_client = None
if USE_GROQ:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        # If SDK isn't installed or init fails, disable Groq gracefully
        groq_client = None
        USE_GROQ = False

# ---------------------------
# Database config
# ---------------------------
DATABASE_URL = "sqlite:///./finance_bot.db"

@st.cache_resource
def get_engine_and_models():
    """
    Create engine and define models inside a cached resource to avoid
    SQLAlchemy table re-definition errors on Streamlit reruns.
    """
    engine = create_engine(DATABASE_URL, echo=False)

    class Expense(SQLModel, table=True):
        __tablename__ = "expense"
        __table_args__ = {"extend_existing": True}

        id: Optional[int] = Field(default=None, primary_key=True)
        user_id: str = Field(default="default_user", index=True)
        amount: float
        category: Optional[str] = "misc"
        note: Optional[str] = None
        created_at: datetime = Field(default_factory=datetime.utcnow)

    class PortfolioItem(SQLModel, table=True):
        __tablename__ = "portfolioitem"
        __table_args__ = {"extend_existing": True}

        id: Optional[int] = Field(default=None, primary_key=True)
        user_id: str = Field(default="default_user", index=True)
        symbol: str
        shares: float
        avg_price: Optional[float] = None
        created_at: datetime = Field(default_factory=datetime.utcnow)

    # Create tables if not exist
    SQLModel.metadata.create_all(engine)
    return engine, Expense, PortfolioItem

engine, Expense, PortfolioItem = get_engine_and_models()

# ---------------------------
# Database helper functions
# ---------------------------
def log_expense(amount: float, category: str, note: Optional[str] = None, user_id: str = "default_user"):
    """Insert expense row and return the created object."""
    with Session(engine) as session:
        e = Expense(user_id=user_id, amount=amount, category=category, note=note)
        session.add(e)
        session.commit()
        session.refresh(e)
        return e

def get_expenses(user_id: str = "default_user") -> pd.DataFrame:
    """Return expenses for user as a DataFrame (possibly empty)."""
    with Session(engine) as session:
        stmt = select(Expense).where(Expense.user_id == user_id).order_by(Expense.created_at.desc())
        rows = session.exec(stmt).all()
    if not rows:
        # Return an empty DataFrame with expected columns so downstream plotting code can rely on them
        return pd.DataFrame(columns=["id", "amount", "category", "note", "created_at"])
    return pd.DataFrame([{
        "id": r.id,
        "amount": r.amount,
        "category": r.category,
        "note": r.note,
        "created_at": r.created_at
    } for r in rows])

def add_portfolio(symbol: str, shares: float, avg_price: Optional[float] = None, user_id: str = "default_user"):
    """Insert a portfolio holding."""
    with Session(engine) as session:
        p = PortfolioItem(user_id=user_id, symbol=symbol.upper(), shares=shares, avg_price=avg_price)
        session.add(p)
        session.commit()
        session.refresh(p)
        return p

def get_portfolio(user_id: str = "default_user") -> pd.DataFrame:
    """Return portfolio holdings as DataFrame (possibly empty)."""
    with Session(engine) as session:
        stmt = select(PortfolioItem).where(PortfolioItem.user_id == user_id)
        rows = session.exec(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["id", "symbol", "shares", "avg_price", "created_at"])
    return pd.DataFrame([{
        "id": r.id,
        "symbol": r.symbol,
        "shares": r.shares,
        "avg_price": r.avg_price,
        "created_at": r.created_at
    } for r in rows])


def risk_profile_from_input(risk_str: str) -> str:
    r = risk_str.lower()
    if "low" in r or "conservative" in r:
        return "low"
    if "high" in r or "aggressive" in r:
        return "high"
    return "medium"

def investment_suggestions(risk_level: str) -> List[str]:
    if risk_level == "low":
        return [
            "High-yield savings account",
            "Government / high-quality corporate bonds",
            "Fixed deposits (term deposits)",
        ]
    if risk_level == "high":
        return [
            "Individual stocks (diversified across sectors)",
            "Small-cap growth stocks",
            "Selective crypto exposure (small % of portfolio)",
        ]
    return [
        "Balanced mutual funds",
        "Index ETFs (broad-market, e.g., S&P 500)",
        "Mixture of bonds and equities",
    ]

def compute_budget_plan(monthly_income: float, monthly_expenses: float) -> Dict[str, float]:
    savings_target = max(0.05 * monthly_income, monthly_income - monthly_expenses)
    essentials = min(monthly_expenses, 0.5 * monthly_income)
    wants = monthly_income - essentials - savings_target
    if wants < 0:
        wants = 0.0
    return {
        "essentials": round(essentials, 2),
        "wants": round(wants, 2),
        "savings": round(savings_target, 2),
    }

def compute_goal_plan(goal_amount: float, months: int, current_savings_per_month: float) -> Dict[str, Any]:
    monthly_needed = goal_amount / months if months > 0 else float("inf")
    shortfall = monthly_needed - current_savings_per_month
    return {
        "monthly_needed": round(monthly_needed, 2),
        "shortfall": round(max(shortfall, 0.0), 2),
        "months": months,
    }

def fetch_live_price(symbol: str) -> Optional[float]:
    """Fetch latest close price using yfinance (optional)."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        t = yf.Ticker(symbol)
        data = t.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        return None
    return None

def ask_groq(prompt: str, system_prompt: str = "You are a helpful, conservative financial assistant.") -> str:
    if not USE_GROQ or not groq_client:
        return "Groq not configured. Set GROQ_API_KEY in .env or Streamlit secrets to enable LLM."
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Finance & Investment Advisor Bot", page_icon="💰", layout="wide")
st.title("💰 Finance & Investment Advisor Bot")

# Sidebar - user inputs and quick actions
with st.sidebar:
    st.header("Profile & Quick Inputs")
    user_name = st.text_input("Your Name (optional)", value="User")
    monthly_income = st.number_input("Monthly Income (PKR)", min_value=0.0, value=50000.0, step=1000.0)
    avg_monthly_expenses = st.number_input("Typical Monthly Expenses (PKR)", min_value=0.0, value=30000.0, step=500.0)
    risk_choice = st.selectbox("Risk appetite", ["Conservative / Low", "Moderate / Medium", "Aggressive / High"])
    selected_risk = risk_profile_from_input(risk_choice)

    st.markdown("---")
    st.subheader("Quick Actions")

    # Expense form
    with st.form("expense_form", clear_on_submit=True):
        e_amt = st.number_input("Expense amount (PKR)", min_value=0.0, step=50.0)
        e_cat = st.selectbox("Category", ["food", "rent", "transport", "utilities", "entertainment", "misc"])
        e_note = st.text_input("Note (optional)")
        if st.form_submit_button("Log Expense"):
            if e_amt > 0:
                log_expense(e_amt, e_cat, e_note)
                st.success(f"Logged expense {e_amt} PKR on {e_cat}")
            else:
                st.warning("Please enter an amount greater than 0.")

    # Portfolio form
    with st.form("portfolio_form", clear_on_submit=True):
        p_symbol = st.text_input("Add portfolio symbol (e.g., AAPL)").upper()
        p_shares = st.number_input("Shares / units", min_value=0.0, step=0.1, value=0.0)
        p_price = st.number_input("Avg price (optional)", min_value=0.0, step=0.01, value=0.0)
        if st.form_submit_button("Add to Portfolio"):
            if p_symbol and p_shares > 0:
                add_portfolio(p_symbol, p_shares, p_price if p_price > 0 else None)
                st.success(f"Added {p_shares} of {p_symbol} to portfolio")
            else:
                st.warning("Please enter a symbol and shares > 0.")

    st.markdown("---")
    if USE_GROQ:
        masked = f"...{GROQ_API_KEY[-4:]}" if isinstance(GROQ_API_KEY, str) and len(GROQ_API_KEY) >= 4 else "(loaded)"
        st.success(f"Groq LLM enabled ({masked})")
    else:
        st.info("Groq LLM: not enabled. Add GROQ_API_KEY to .env or .streamlit/secrets.toml")

# Main content
st.header(f"Welcome, {user_name} — Your personalized plan")

# Budget suggestion
st.subheader("🔎 Quick Budget Suggestion")
budget = compute_budget_plan(monthly_income, avg_monthly_expenses)
st.write(f"Essentials: {budget['essentials']} PKR / month")
st.write(f"Wants (discretionary): {budget['wants']} PKR / month")
st.write(f"Recommended savings target: {budget['savings']} PKR / month")

# Investment suggestions
st.subheader("📈 Investment Suggestions (risk-based)")
st.write(f"Risk level: **{selected_risk.capitalize()}**")
for s in investment_suggestions(selected_risk):
    st.markdown(f"- {s}")

# Goal planner
st.subheader("🎯 Goal Planner")
with st.form("goal_form"):
    goal_name = st.text_input("Goal name (e.g., Car, House, Emergency Fund)", value="Car")
    goal_amount = st.number_input("Goal amount (PKR)", min_value=0.0, value=500000.0, step=1000.0)
    goal_years = st.number_input("Time horizon (years)", min_value=0, max_value=50, value=5)
    if st.form_submit_button("Calculate Plan"):
        months = int(goal_years * 12)
        plan = compute_goal_plan(goal_amount, months if months > 0 else 1, budget["savings"])
        st.write(f"To reach **{goal_name}** of {goal_amount} PKR in {months} months:")
        st.write(f"- You need to save **{plan['monthly_needed']} PKR / month**")
        if plan["shortfall"] > 0:
            st.warning(f"- Shortfall of **{plan['shortfall']} PKR / month** vs current recommended savings ({budget['savings']} PKR). Consider increasing income or reducing expenses.")
        else:
            st.success("You're on track with current recommended savings!")

# Expenses display & visualization
st.subheader("🧾 Expenses")
df_exp = get_expenses()
if df_exp.empty or df_exp.shape[0] == 0:
    st.info("No expenses logged yet. Use the sidebar to log quick expenses.")
else:
    # Show recent expenses
    st.write("Recent expenses:")
    st.dataframe(df_exp[["created_at", "amount", "category", "note"]].head(100))

    # Prepare date column
    df_exp["created_date"] = pd.to_datetime(df_exp["created_at"]).dt.date

    # Use current month by default (fallback to all data)
    month_start = date.today().replace(day=1)
    recent = df_exp[df_exp["created_date"] >= month_start]
    if recent.empty:
        recent = df_exp.copy()

    # Category sum for pie chart (guard empty)
    if not recent.empty:
        cat_sum = recent.groupby("category", dropna=False)["amount"].sum().reset_index()
    else:
        cat_sum = pd.DataFrame(columns=["category", "amount"])

    if not cat_sum.empty and cat_sum["amount"].sum() > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.pie(cat_sum["amount"], labels=cat_sum["category"], autopct="%1.1f%%")
        ax1.set_title("Category share (current month)")
        st.pyplot(fig1)
    else:
        st.info("No spend distribution to plot yet (no amounts in the selected period).")

    # Avg monthly vs savings (rough)
    if not df_exp.empty:
        ym = pd.to_datetime(df_exp["created_at"]).dt.to_period("M")
        months_count = max(1, ym.nunique())
        monthly_total = float(df_exp["amount"].sum() / months_count)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(["Avg Monthly Expenses", "Savings Target"], [monthly_total, budget["savings"]])
        ax2.set_ylabel("PKR")
        st.pyplot(fig2)

    # CSV export
    st.download_button(
        "Download expenses CSV",
        df_exp.to_csv(index=False).encode("utf-8"),
        file_name="expenses.csv",
        mime="text/csv",
    )

# Portfolio section
st.subheader("📂 Portfolio")
df_port = get_portfolio()
if df_port.empty or df_port.shape[0] == 0:
    st.info("Portfolio empty. Add holdings from the sidebar.")
else:
    st.dataframe(df_port)
    if YFINANCE_AVAILABLE:
        st.write("Fetching live prices (yfinance)...")
        values = []
        for _, row in df_port.iterrows():
            symbol = row["symbol"]
            price = fetch_live_price(symbol)
            market_price = price if price is not None else (row["avg_price"] if row["avg_price"] else 0.0)
            try:
                value = float(market_price) * float(row["shares"])
            except Exception:
                value = 0.0
            values.append({
                "symbol": symbol,
                "shares": row["shares"],
                "market_price": round(float(market_price), 2),
                "value": round(value, 2),
            })
        if values:
            df_values = pd.DataFrame(values)
            st.dataframe(df_values)
            st.write(f"Total portfolio value: {df_values['value'].sum()} PKR")
        else:
            st.info("No priced holdings to display yet.")
    else:
        st.info("Live prices disabled (yfinance not installed). Install with: pip install yfinance")

# Chat / Q&A section
st.subheader("💬 Ask the bot (questions, planning help, quick calculations)")
with st.form("qa_form", clear_on_submit=False):
    user_q = st.text_area(
        "Type your question (e.g., 'How much should I save monthly for a 200k car in 3 years?')",
        height=120,
    )
    qa_submit = st.form_submit_button("Ask")
    if qa_submit and user_q.strip():
        if USE_GROQ and groq_client:
            st.write("Generating answer (Groq LLM)...")
            answer = ask_groq(
                user_q,
                system_prompt="You are a friendly, conservative financial assistant. Provide clear actionable steps and simple calculations. Keep answers concise."
            )
            st.markdown(answer)
        else:
            q = user_q.lower()
            import re
            if "save" in q and "in" in q and ("years" in q or "year" in q):
                m_amount = re.search(r"(\d[\d,\.]*)", q.replace(",", ""))
                m_years = re.search(r"(\d+)\s*(year|years)", q)
                if m_amount and m_years:
                    try:
                        amt = float(m_amount.group(1))
                        yrs = int(m_years.group(1))
                        months = yrs * 12
                        monthly = round(amt / months, 2) if months > 0 else None
                        st.write(f"To save {amt:.0f} in {yrs} years, you need about **{monthly:.0f} PKR / month**.")
                    except Exception:
                        st.write("I couldn't compute that amount. Try: 'Save 500000 in 5 years'.")
                else:
                    st.write("Sorry, I couldn't parse your question. Example: 'Save 500000 in 5 years'.")
            elif "invest" in q or "where to invest" in q or "risk" in q:
                st.write(f"Based on your risk: **{selected_risk}**")
                for s in investment_suggestions(selected_risk):
                    st.markdown(f"- {s}")
            else:
                st.write("Groq not enabled and I couldn't match a pattern. Try a simpler question or enable Groq LLM.")

st.markdown("---")
st.info("Tip: Use the sidebar to log expenses and add portfolio items. Put your GROQ_API_KEY in .env or .streamlit/secrets.toml to enable LLM features.")