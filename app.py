import streamlit as st
import pandas as pd
from datetime import datetime, date
import io
import os
import pytz
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF


@st.fragment
def search_vault(df_hist_master):
    # This block can rerun independently without reloading the simulation
    with st.form("vault_search_form"):
        t_filt = st.text_input("Target Search")
        if st.form_submit_button("🔍 Run Search"):
            # Update specific session state
            st.session_state.filtered_vault = df_hist_master[df_hist_master["Target"].str.contains(t_filt)]
# ==========================================
# PDF ENGINE (FIXED KEYS)
# ==========================================
def create_pdf(rec, kpi_data, sources_uses, amort_schedule):
    from fpdf import FPDF
    from datetime import datetime

    # 1. Define a custom PDF class to handle Footer and Page Numbers
    class PDF(FPDF):
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            # Confidentiality Disclaimer
            self.cell(0, 10, 'CONFIDENTIAL - FOR STRATEGIC REVIEW ONLY', 0, 0, 'L')
            # Page Number
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')

    pdf = PDF() # Use our custom class
    pdf.alias_nb_pages() # Allows for "Page 1 of X" logic if needed
    pdf.add_page()
    
    # --- HEADER ---
    pdf.set_fill_color(31, 73, 125) 
    pdf.set_text_color(255, 255, 255) 
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "STRATEGIC ACQUISITION SUMMARY", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(2)

    # --- 1. IDENTITY & LOCATION SECTION ---
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 1. Target Identity & Location", ln=True, fill=True)
    pdf.set_font("Arial", '', 10); pdf.ln(2)
    
    pdf.cell(95, 7, f"Target Name: {str(rec.get('Target', 'N/A'))}")
    pdf.cell(95, 7, f"Location: {str(rec.get('City', 'N/A'))}, {str(rec.get('State', 'N/A'))}", ln=True)
    pdf.ln(2)

    # --- THE VERDICT BANNER BOX ---
    verdict_str = str(rec.get('Verdict', 'N/A')).upper()
    if "PASS" in verdict_str:
        pdf.set_fill_color(39, 174, 96); pdf.set_text_color(255, 255, 255)
    elif "CAUTION" in verdict_str:
        pdf.set_fill_color(241, 196, 15); pdf.set_text_color(0, 0, 0)
    elif "FAIL" in verdict_str:
        pdf.set_fill_color(231, 76, 60); pdf.set_text_color(255, 255, 255)
    else:
        pdf.set_fill_color(200, 200, 200); pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f" VERDICT: {verdict_str}", ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # --- 2. PERFORMANCE KPIs ---
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 2. Key Performance Indicators (Year 1)", ln=True, fill=True)
    pdf.set_font("Arial", '', 9); pdf.ln(2)
    for label, value in kpi_data.items():
        if label != "dscr":
            pdf.cell(110, 7, f" {str(label)}", border=1)
            pdf.cell(80, 7, f" {str(value)}", border=1, ln=True, align='C')
    pdf.ln(4)

    # --- 3. SOURCES & USES ---
    if sources_uses:
        pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, " 3. Deal Financing Structure (Sources & Uses)", ln=True, fill=True); pdf.ln(2)
        pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(220, 220, 220)
        pdf.cell(45, 7, "Sources", 1, 0, 'C', 1); pdf.cell(50, 7, "Amount", 1, 0, 'C', 1)
        pdf.cell(45, 7, "Uses", 1, 0, 'C', 1); pdf.cell(50, 7, "Amount", 1, 1, 'C', 1)
        pdf.set_font("Arial", '', 9)
        s_items, u_items = sources_uses.get('sources', []), sources_uses.get('uses', [])
        for i in range(max(len(s_items), len(u_items))):
            s_l = s_items[i][0] if i < len(s_items) else ""; s_v = s_items[i][1] if i < len(s_items) else ""
            u_l = u_items[i][0] if i < len(u_items) else ""; u_v = u_items[i][1] if i < len(u_items) else ""
            pdf.cell(45, 6, f" {s_l}", 1); pdf.cell(50, 6, f" {s_v}", 1, 0, 'R')
            pdf.cell(45, 6, f" {u_l}", 1); pdf.cell(50, 6, f" {u_v}", 1, 1, 'R')
        pdf.ln(6)

    # --- 4. STRATEGIC SUMMARY ---
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 4. Executive Strategic Rationale", ln=True, fill=True)
    pdf.ln(2); pdf.set_font("Arial", '', 9)
    reasoning = str(rec.get('Reasons', 'No notes provided.')).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, reasoning)

    # --- AMORTIZATION SCHEDULES (Appendix A & B) ---
    if amort_schedule:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Appendix A: Bank Loan Amortization Schedule", ln=True); pdf.ln(2)

        t_b_p = t_b_i = t_b_pay = 0
        pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(225, 245, 254)
        pdf.cell(20, 8, "Year", 1, 0, 'C', 1); pdf.cell(40, 8, "Principal", 1, 0, 'C', 1)
        pdf.cell(40, 8, "Interest", 1, 0, 'C', 1); pdf.cell(40, 8, "Total Payment", 1, 0, 'C', 1)
        pdf.cell(40, 8, "Balance", 1, 1, 'C', 1)

        pdf.set_font("Arial", '', 9)
        for row in amort_schedule:
            if row['Year'] == "CUMULATIVE TOTALS": continue
            if row.get('Bank Payment', 0) > 0:
                pdf.cell(20, 7, str(row['Year']), 1, 0, 'C')
                pdf.cell(40, 7, f"{row.get('Bank Principal', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Bank Interest', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Bank Payment', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Bank Balance', 0):,.0f}", 1, 1, 'R')
                t_b_p += row.get('Bank Principal', 0); t_b_i += row.get('Bank Interest', 0); t_b_pay += row.get('Bank Payment', 0)

        pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(240, 240, 240)
        pdf.cell(20, 8, "TOTAL", 1, 0, 'C', 1); pdf.cell(40, 8, f"{t_b_p:,.0f}", 1, 0, 'R', 1)
        pdf.cell(40, 8, f"{t_b_i:,.0f}", 1, 0, 'R', 1); pdf.cell(40, 8, f"{t_b_pay:,.0f}", 1, 0, 'R', 1); pdf.cell(40, 8, "-", 1, 1, 'C', 1)

        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Appendix B: Seller Note Amortization Schedule", ln=True); pdf.ln(2)

        t_s_p = t_s_i = t_s_pay = 0
        pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(255, 243, 224)
        pdf.cell(20, 8, "Year", 1, 0, 'C', 1); pdf.cell(40, 8, "Principal", 1, 0, 'C', 1)
        pdf.cell(40, 8, "Interest", 1, 0, 'C', 1); pdf.cell(40, 8, "Total Payment", 1, 0, 'C', 1)
        pdf.cell(40, 8, "Balance", 1, 1, 'C', 1)

        pdf.set_font("Arial", '', 9)
        for row in amort_schedule:
            if row['Year'] == "CUMULATIVE TOTALS": continue
            if row.get('Seller Payment', 0) > 0:
                pdf.cell(20, 7, str(row['Year']), 1, 0, 'C')
                pdf.cell(40, 7, f"{row.get('Seller Principal', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Seller Interest', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Seller Payment', 0):,.0f}", 1, 0, 'R')
                pdf.cell(40, 7, f"{row.get('Seller Balance', 0):,.0f}", 1, 1, 'R')
                t_s_p += row.get('Seller Principal', 0); t_s_i += row.get('Seller Interest', 0); t_s_pay += row.get('Seller Payment', 0)

        pdf.set_font("Arial", 'B', 9); pdf.set_fill_color(240, 240, 240)
        pdf.cell(20, 8, "TOTAL", 1, 0, 'C', 1); pdf.cell(40, 8, f"{t_s_p:,.0f}", 1, 0, 'R', 1)
        pdf.cell(40, 8, f"{t_s_i:,.0f}", 1, 0, 'R', 1); pdf.cell(40, 8, f"{t_s_pay:,.0f}", 1, 0, 'R', 1); pdf.cell(40, 8, "-", 1, 1, 'C', 1)

    # --- PAGE 3: CONSOLIDATED SUMMARY & VERDICT ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Appendix C: Consolidated Financing Summary", ln=True); pdf.ln(5)

    pdf.set_fill_color(240, 240, 240); pdf.set_font("Arial", 'B', 10)
    pdf.cell(100, 10, "Metric", 1, 0, 'L', 1); pdf.cell(80, 10, "Total Amount", 1, 1, 'C', 1)
    
    pdf.set_font("Arial", '', 10)
    summary_data = [
        ("Total Combined Principal", t_b_p + t_s_p),
        ("Total Interest Expense", t_b_i + t_s_i),
        ("Total Cash Outlay (Debt Service)", t_b_pay + t_s_pay)
    ]
    for metric, value in summary_data:
        pdf.cell(100, 10, f" {metric}", 1); pdf.cell(80, 10, f"${value:,.0f}", 1, 1, 'R')

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Financial Health Assessment", ln=True); pdf.ln(2)

    dscr_v = kpi_data.get('dscr', 0)
    if dscr_v >= 1.25:
        assess = "HEALTHY: Strong coverage of debt obligations."
        pdf.set_fill_color(232, 245, 233); pdf.set_text_color(46, 125, 50)
    elif dscr_v >= 1.0:
        assess = "TIGHT: Debt is covered but offers minimal safety margin."
        pdf.set_fill_color(255, 248, 225); pdf.set_text_color(245, 127, 23)
    else:
        assess = "CRITICAL: Insufficient cash flow for debt service."
        pdf.set_fill_color(255, 235, 238); pdf.set_text_color(198, 40, 40)

    pdf.set_font("Arial", 'B', 11)
    pdf.multi_cell(0, 12, f"  {assess}", border=1, fill=True)

    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output)

# ==========================================
# APP CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Magna Vita Financial Dashboard", page_icon="📈")

@st.cache_data
def load_data_final():
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    subsidiaries = ["All About Caring - Rush City", "All About Caring - Twin Cities", "Communities of Care"]
    service_types = ["PCA", "Complex Nursing", "Other Services"]
    data = []
    for sub in subsidiaries:
        base_annual_rev = np.random.randint(600000, 1000000)
        base_daily_rev = base_annual_rev / 365
        base_emp = np.random.randint(20, 50) 
        base_cli = np.random.randint(50, 100) 
        for d in dates:
            rev = float((base_daily_rev * (0.5 if d.weekday() >= 5 else 1.1)) * np.random.uniform(0.8, 1.2))
            cogs = float(rev * np.random.uniform(0.4, 0.5))
            opex = float(rev * np.random.uniform(0.2, 0.3))
            data.append({
                "Date": d, "Year": int(d.year), "Subsidiary": str(sub), "Service Type": np.random.choice(service_types),
                "Revenue": rev, "COGS": cogs, "Expenses": opex, "EBITDA": (rev - cogs) - opex,
                "Gross Margin": rev - cogs, "Client Count": float(base_cli), "Employee Count": float(base_emp),
                "New Clients": float(np.random.poisson(0.1)), "Timestamp_DT": d
            })
    return pd.DataFrame(data)

df_master = load_data_final()

# Custom CSS for UI Improvements
st.markdown("""
    <style>
    /* 1. GENERAL PADDING */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    @media print {
        /* Force 1-page layout by shrinking content */
        html, body, [data-testid="stAppViewContainer"] {
            zoom: 85%; 
            background-color: white !important;
        }
        /* Hide sidebars, headers, and navigation from the PDF */
        [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stDecoration"], .stRadio {
            display: none !important;
        }
        .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
        }
        /* Ensure no page breaks inside charts */
        .element-container, .stPlotlyChart {
            page-break-inside: avoid !important;
        }
    }
                    
    /* 2. SIDEBAR LAYOUT STYLING */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem; 
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: flex-start;
    }
    
    [data-testid="stSidebarUserContent"] div:has(p) {
        margin-top: auto;
        padding-bottom: 20px;
    }

    /* 3. METRIC CARD STYLING */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4e8cff;
    }
    
    /* 4. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* 5. CUSTOM HEADERS */
    .step-header {
        font-size: 1.0rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    /* 6. ACQUISITION SIMULATION CUSTOM STYLES */
    .memo-box {
        background-color: #f1f3f5;
        padding: 25px;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.4;
    }
    .section-header-blue {
        background-color: #4e8cff;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (SIMULATION MODE)
# ==========================================
@st.cache_data
def load_data_final():
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    
    subsidiaries = [
        "All About Caring - Rush City", 
        "All About Caring - Twin Cities", 
        "Communities of Care"
    ]
    
    service_types = [
        "PCA", 
        "Complex Nursing", 
        "Other Services"
    ]
    
    data = []
    for sub in subsidiaries:
        base_annual_rev = np.random.randint(600000, 1000000)
        base_daily_rev = base_annual_rev / 365
        
        base_emp = np.random.randint(20, 50) 
        base_cli = np.random.randint(50, 100) 
        
        for d in dates:
            is_weekend = d.weekday() >= 5
            daily_factor = 0.5 if is_weekend else 1.1
            
            seasonality = 1 + (0.1 * np.sin(2 * np.pi * d.dayofyear / 365))
            rev = float(base_daily_rev * daily_factor * seasonality * np.random.uniform(0.8, 1.2))
            
            cogs = float(rev * np.random.uniform(0.4, 0.5))
            opex = float(rev * np.random.uniform(0.2, 0.3))
            marketing_spend = float(rev * np.random.uniform(0.05, 0.15)) 
            
            contact_hrs = float(np.random.randint(30, 70)) if not is_weekend else float(np.random.randint(0, 10))
            sched_hrs = float(contact_hrs * np.random.uniform(1.05, 1.15))
            qual_hrs = float(sched_hrs * np.random.uniform(1.1, 1.2))
            bill_hrs = float(contact_hrs * np.random.uniform(0.95, 1.0))
            
            emp_count = float(int(base_emp * np.random.uniform(0.99, 1.01)))
            client_count = float(int(base_cli * np.random.uniform(0.99, 1.01)))
            
            new_clients = float(np.random.poisson(0.1)) 
            lost_clients = float(np.random.poisson(0.03)) 
            new_hires = float(np.random.poisson(0.03)) 
            departures = float(np.random.poisson(0.015))

            rec_spend = (new_hires * 1500.0) + np.random.uniform(50, 200)
            
            leads_contacted = new_clients * np.random.randint(3, 6) if new_clients > 0 else 0
            sales_cycle_days = np.random.randint(14, 45) if new_clients > 0 else 0
            avg_contract_days = float(np.random.uniform(280, 365))

            svc = np.random.choice(service_types)

            data.append({
                "Date": d,
                "Year": int(d.year),
                "Quarter": int(d.quarter),
                "Month": int(d.month),
                "Week": int(d.isocalendar()[1]),
                "Subsidiary": str(sub),
                "Service Type": svc,
                "Revenue": rev,
                "COGS": cogs,
                "Expenses": opex,
                "Marketing Spend": marketing_spend,
                "Recruitment Spend": rec_spend, 
                "Gross Margin": rev - cogs,
                "EBITDA": (rev - cogs) - opex,
                "D&A": rev * 0.05,
                "Contact Hours": contact_hrs,
                "Scheduled Hours": sched_hrs,
                "Qualified Hours": qual_hrs,
                "Billable Hours": bill_hrs,
                "Employee Count": emp_count,
                "Client Count": client_count,
                "New Clients": new_clients,
                "Lost Clients": lost_clients,
                "New Hires": new_hires,
                "Departures": departures,
                "Leads Contacted": leads_contacted,
                "Avg Sales Cycle": sales_cycle_days,
                "Avg Contract Duration": avg_contract_days
            })
            
    df = pd.DataFrame(data)
    return df

df_master = load_data_final()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def plot_line_chart(df, x, y, title, color=None):
    if y not in df.columns:
        return go.Figure().update_layout(title=f"Error: Column {y} missing")
    fig = px.line(df, x=x, y=y, title=title, color=color, markers=True)
    fig.update_layout(xaxis_title="", yaxis_title=y, template="plotly_white", hovermode="x unified")
    return fig

def get_historical_avg(df, sub, metric, freq='Monthly', year=None, service_types=None):
    numeric_df = df.copy()
    if sub != "Magna Vita (Consolidated)":
        numeric_df = numeric_df[numeric_df["Subsidiary"] == sub]
    if service_types and len(service_types) > 0:
        numeric_df = numeric_df[numeric_df["Service Type"].isin(service_types)]
    if year:
        numeric_df = numeric_df[numeric_df["Year"] == year]

    if freq == 'Monthly':
        return numeric_df.groupby(['Year', 'Month'])[metric].sum().mean()
    elif freq == 'Quarterly':
        return numeric_df.groupby(['Year', 'Quarter'])[metric].sum().mean()
    elif freq == 'Annual':
        return numeric_df.groupby(['Year'])[metric].sum().mean()
    elif freq == 'Weekly':
        return numeric_df.groupby(['Year', 'Week'])[metric].sum().mean()
    return 0.0

# ==========================================
# 4. VIEW: HOME
# ==========================================
def show_home():
    # Detect System Clock (Adjust to user's local timezone)
    user_tz = pytz.timezone('America/New_York') 
    now = datetime.now(user_tz)

    st.markdown("""
        <style>
        [data-testid="stImage"] { margin-top: 20px; }
        </style>
    """, unsafe_allow_html=True)

    c_logo, c_rest = st.columns([1, 9]) 
    with c_logo:
        try:
            st.image("MagnaVitaLogo.jpeg", width=100) 
        except Exception:
            st.markdown("### 🏥")

    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.markdown("""
            <h1 style='text-align: center;'>
                Magna Vita Financial Dashboard <span style='color: #4e8cff; font-size: 0.8em;'>(BETA)</span>
            </h1>
        """, unsafe_allow_html=True)
        st.markdown(f"""
            <p style='text-align: center; color: gray;'>
                <b>Date Built:</b> {date.today().strftime('%B %d, %Y')}<br>
                <b>Creator:</b> Amanda Zheng
            </p>
            """, unsafe_allow_html=True)
        
        st.success("👋 Welcome! Use the sidebar to navigate between Historical Performance, Financial Projections, and the Acquisition Simulation.")
        with st.expander("📝 Read Me: Assumptions & Notes", expanded=True):
            st.markdown("""
            * **Data Source:** Modeled data based on 2024-2025 financial trends.
            * **Granularity:** Daily data aggregated to Weekly, Monthly, or Quarterly views.
            * **Integration:** Designed to eventually integrate with *AlayaCare* API.
            * **Missing Data:** Imputed using historical averages where actuals are missing.
            """)

# ==========================================
# 5. VIEW: HISTORICAL PERFORMANCE (FIXED EXECUTIVE SUMMARY SYNTAX)
# ==========================================
def show_history():
    st.title("🏛️ Historical Performance")
    
    # --- A. FILTERS ---
    with st.expander("🔎 Filter & Baseline Comparison Settings", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sub_select = st.multiselect("Select Subsidiary", options=df_master["Subsidiary"].unique(), default=df_master["Subsidiary"].unique())
        with c2:
            service_select = st.multiselect("Service Type", options=sorted(df_master["Service Type"].unique()), default=df_master["Service Type"].unique())
        with c3:
            baseline_date = st.date_input("Baseline Comparison Date", value=date.today(), max_value=date.today())

        c4, c5, c6 = st.columns(3)
        with c4:
            view_by = st.radio("Group By", ["Week", "Month", "Quarter", "Year"], horizontal=True)
        with c5:
            comparison_mode = st.radio("Comparison Mode", ["Previous Period", "Year-over-Year (YoY)"], horizontal=True)
        with c6:
            available_years = sorted(df_master["Year"].unique())
            target_default_year = baseline_date.year
            safe_default = [target_default_year] if target_default_year in available_years else [available_years[-1]]
            year_select = st.multiselect("Years in Charts", options=available_years, default=safe_default)

    # --- B. DATA PREPARATION ---
    df_prep = df_master.copy()
    if "Date" not in df_prep.columns:
        df_prep["Date"] = pd.to_datetime(df_prep[['Year', 'Month']].assign(Day=1))
    
    df_filt = df_prep[(df_prep["Subsidiary"].isin(sub_select)) & (df_prep["Service Type"].isin(service_select)) & (df_prep["Date"].dt.date <= baseline_date)]
    
    if df_filt.empty:
        st.warning("⚠️ No data found for the selected filters.")
        return

    agg_rules = {
        "Revenue": "sum", "COGS": "sum", "Expenses": "sum", "Gross Margin": "sum", 
        "EBITDA": "sum", "Contact Hours": "sum", "Scheduled Hours": "sum", 
        "Qualified Hours": "sum", "Billable Hours": "sum",
        "New Hires": "sum", "Departures": "sum", "D&A": "sum",
        "Marketing Spend": "sum", "Recruitment Spend": "sum", 
        "New Clients": "sum", "Lost Clients": "sum",
        "Leads Contacted": "sum", "Avg Sales Cycle": "mean",
        "Avg Contract Duration": "mean", "Employee Count": "mean", "Client Count": "mean"
    }
    
    group_cols = ["Year", "Week"] if view_by == "Week" else ["Year", "Month"] if view_by == "Month" else ["Year", "Quarter"] if view_by == "Quarter" else ["Year"]
    df_raw = df_filt.groupby(group_cols).agg(agg_rules).reset_index().sort_values(group_cols)

    def process_ratios(df):
        d = df.copy()
        d["Net Income"] = (d["EBITDA"] - d["D&A"]) * 0.75
        d["Gross Margin %"] = (d["Gross Margin"] / d["Revenue"] * 100).fillna(0)
        d["EBITDA %"] = (d["EBITDA"] / d["Revenue"] * 100).fillna(0)
        d["Net Margin %"] = (d["Net Income"] / d["Revenue"] * 100).fillna(0)
        d["COGS %"] = (d["COGS"] / d["Revenue"] * 100).fillna(0)
        d["OpEx %"] = (d["Expenses"] / d["Revenue"] * 100).fillna(0)
        d["Rev per Client"] = (d["Revenue"] / d["Client Count"]).fillna(0)
        d["Rev per Caregiver"] = (d["Revenue"] / d["Employee Count"]).fillna(0)
        d["Contact Hours per Client"] = (d["Contact Hours"] / d["Client Count"]).fillna(0)
        d["Recruitment / Rev %"] = (d["Recruitment Spend"] / d["Revenue"] * 100).fillna(0)
        d["Recruitment / Rev per CG %"] = (d["Recruitment Spend"] / d["Rev per Caregiver"] * 100).fillna(0)
        d["Service Efficiency %"] = (d["Scheduled Hours"] / d["Qualified Hours"] * 100).fillna(0)
        d["Scheduling Efficiency %"] = (d["Contact Hours"] / d["Scheduled Hours"] * 100).fillna(0)
        d["Billing Efficiency %"] = (d["Billable Hours"] / d["Contact Hours"] * 100).fillna(0)
        d["CAC"] = (d["Marketing Spend"] / d["New Clients"]).replace([np.inf, -np.inf], 0).fillna(0)
        d["CAC / Rev per Client %"] = (d["CAC"] / d["Rev per Client"] * 100).fillna(0)
        d["CAC / COGS per Client %"] = (d["CAC"] / (d["COGS"] / d["Client Count"]) * 100).fillna(0)
        d["Sales Conv %"] = (d["New Clients"] / d["Leads Contacted"] * 100).fillna(0)
        
        if view_by == "Week": d["Period"] = d.apply(lambda x: f"{int(x['Year'])}-W{int(x['Week']):02d}", axis=1)
        elif view_by == "Month": d["Period"] = d.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
        elif view_by == "Quarter": d["Period"] = d.apply(lambda x: f"{int(x['Year'])}-Q{int(x['Quarter'])}", axis=1)
        else: d["Period"] = d["Year"].astype(str)
        return d

    df_view_all = process_ratios(df_raw)
    df_view = df_view_all[df_view_all["Year"].isin(year_select)]

    # --- C. KPIs & COMPARISON ---
    curr = df_view_all.iloc[-1]
    prev = None
    comp_label = "No Comparison Data"
    if comparison_mode == "Year-over-Year (YoY)":
        mask = (df_view_all["Year"] == curr["Year"]-1)
        for col in [c for c in group_cols if c != "Year"]: mask &= (df_view_all[col] == curr[col])
        match = df_view_all[mask]
        if not match.empty:
            prev = match.iloc[0]
            comp_label = f"vs Same {view_by} {int(curr['Year']-1)}"
    elif len(df_view_all) >= 2:
        prev = df_view_all.iloc[-2]
        comp_label = f"vs Previous {view_by}"

    def get_delta(field, is_money=False, is_pct=False):
        if prev is None: return None
        diff = curr[field] - prev[field]
        if is_pct: return f"{diff:.1f}%"
        return f"${diff:,.0f}" if is_money else f"{diff:,.0f}"

    # --- D. KPI DISPLAY (2 ROWS OF 4) ---
    st.markdown(f"### 📊 Analysis for Period: {curr['Period']} ({comp_label})")
    
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    r1_c1.metric("Revenue", f"${curr['Revenue']:,.0f}", delta=get_delta('Revenue', True))
    r1_c2.metric("EBITDA", f"${curr['EBITDA']:,.0f}", delta=get_delta('EBITDA', True))
    r1_c3.metric("EBITDA %", f"{curr['EBITDA %']:.1f}%", delta=get_delta('EBITDA %', False, True))
    r1_c4.metric("Net Income", f"${curr['Net Income']:,.0f}", delta=get_delta('Net Income', True))
    
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    r2_c1.metric("Contact Hours", f"{curr['Contact Hours']:,.0f}", delta=get_delta('Contact Hours'))
    r2_c2.metric("Active Clients", f"{curr['Client Count']:,.0f}", delta=get_delta('Client Count'))
    r2_c3.metric("Employee Count", f"{curr['Employee Count']:,.0f}", delta=get_delta('Employee Count'))
    r2_c4.metric("Net Margin %", f"{curr['Net Margin %']:.1f}%", delta=get_delta('Net Margin %', False, True))

    # --- E. EXECUTIVE PERFORMANCE SUMMARY (ENHANCED STRATEGIC INSIGHTS) ---
    if prev is not None:
        st.markdown("---")
        st.subheader("📝 Strategic Executive Summary")
        
        # 1. Calculation Engine for Advanced Insights
        def get_pct_change(field):
            return ((curr[field] / prev[field]) - 1) * 100 if prev[field] != 0 else 0

        rev_pct = get_pct_change('Revenue')
        ebitda_pct = get_pct_change('EBITDA')
        hours_pct = get_pct_change('Contact Hours')
        margin_delta = curr['EBITDA %'] - prev['EBITDA %']
        
        # 2. Portfolio-Wide Logic (Conditional)
        all_subs = df_master["Subsidiary"].unique()
        is_all_selected = len(sub_select) == len(all_subs)
        
        # 3. New Insight: Unit Economics (LTV/CAC proxy)
        rev_per_client_delta = get_pct_change('Rev per Client')
        cac_delta = get_pct_change('CAC')
        unit_econ_msg = ""
        if rev_per_client_delta > 0 and cac_delta < 0:
            unit_econ_msg = "✅ **Efficiency Gain:** Revenue per client is rising while acquisition costs are falling."
        elif cac_delta > rev_per_client_delta:
            unit_econ_msg = "⚠️ **Margin Pressure:** Customer acquisition costs (CAC) are outpacing revenue growth per client."

        # 4. New Insight: Operational Throughput
        billing_eff = curr['Billing Efficiency %']
        sched_eff = curr['Scheduling Efficiency %']
        ops_msg = f"Operational throughput is at {billing_eff:.1f}%. "
        if billing_eff < sched_eff:
            ops_msg += "There is a significant leak between scheduled and billable hours that requires audit."

        # 5. Subsidiary & Workforce (All-Selected Logic)
        spotlight_section = ""
        workforce_warning = ""
        if is_all_selected:
            df_spotlight = df_prep[(df_prep["Year"] == curr["Year"]) & (df_prep["Month"] == curr["Month"] if "Month" in curr else True)].copy()
            if not df_spotlight.empty:
                df_spotlight["Margin"] = (df_spotlight["EBITDA"] / df_spotlight["Revenue"]).fillna(0)
                best_sub = df_spotlight.loc[df_spotlight["Margin"].idxmax(), "Subsidiary"]
                worst_sub = df_spotlight.loc[df_spotlight["Margin"].idxmin(), "Subsidiary"]
                spotlight_section = f"* 🌟 **Top Performer:** {best_sub} | ⚠️ **Underperformer:** {worst_sub}"
            
            cg_ratio = curr['Employee Count'] / curr['Client Count'] if curr['Client Count'] > 0 else 0
            if cg_ratio < (prev['Employee Count'] / prev['Client Count']) * 0.95:
                workforce_warning = f"🚨 **Critical Capacity Alert:** Staffing levels per client have dropped by >5%. Burnout risk is elevated."

        # 6. Final Narrative Construction
        narrative_text = f"""
        **Executive Overview:** {curr['Period']} showed a **{rev_pct:.1f}%** revenue shift and a **{ebitda_pct:.1f}%** change in EBITDA. 
        The portfolio is currently operating at a **{curr['EBITDA %']:.1f}%** margin ({margin_delta:+.1f} bps vs {comparison_mode}).

        **Strategic Drivers & Unit Economics:**
        * {unit_econ_msg if unit_econ_msg else "Unit economics remain stable relative to the prior period."}
        * **Volume Analysis:** Contact hours moved by **{hours_pct:.1f}%**. {ops_msg}
        
        **HR & Capacity:**
        * Recruitment spend is **{curr['Recruitment / Rev %']:.2f}%** of revenue. 
        * {workforce_warning if workforce_warning else "Staffing-to-client ratios remain within optimal operational bands."}

        **Portfolio Spotlight:**
        {spotlight_section if spotlight_section else "Detailed subsidiary analysis is available in the individual filters."}
        """

        

        if ebitda_pct > 0:
            st.success(narrative_text)
        else:
            st.info(narrative_text)

    # --- F. TABS & ALL RESTORED CHARTS ---
    tabs = st.tabs(["💰 Revenue", "📈 Profitability", "📉 Cost Analysis", "⚙️ Operations", "👥 Human Resource", "🤝 Clients", "📢 Marketing & Sales Metrics", "📥 Raw Data"])

    with tabs[0]: # Revenue
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Revenue", "Total Revenue Trend"), use_container_width=True)
        with c2: 
            # Sub-chart for Service breakdown
            df_svc = df_filt.groupby(group_cols + ["Service Type"])["Revenue"].sum().reset_index()
            # Simple period labeling for chart
            df_svc["Period"] = df_svc["Year"].astype(str) + (("-" + df_svc["Month"].astype(str)) if "Month" in df_svc else "")
            st.plotly_chart(px.bar(df_svc, x="Period", y="Revenue", color="Service Type", title="Revenue Proportion", barmode='stack'), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Rev per Client", "Avg Revenue per Client"), use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Rev per Caregiver", "Avg Revenue per Caregiver"), use_container_width=True)

    with tabs[1]: # Profitability
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "EBITDA", "EBITDA Trend"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Net Income", "Net Income Trend"), use_container_width=True)
        fig_p = go.Figure()
        for m in ["Gross Margin %", "EBITDA %", "Net Margin %"]: fig_p.add_trace(go.Scatter(x=df_view['Period'], y=df_view[m], name=m))
        fig_p.update_layout(title="Margin Analysis Over Time", template="plotly_white")
        st.plotly_chart(fig_p, use_container_width=True)

    with tabs[2]: # Cost Analysis
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "COGS", "Total COGS ($)"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "COGS %", "COGS % of Revenue"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Expenses", "Total OpEx ($)"), use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "OpEx %", "OpEx % of Revenue"), use_container_width=True)

    with tabs[3]: # Operations
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Service Efficiency %", "Service Eff %"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Scheduling Efficiency %", "Sched Eff %"), use_container_width=True)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Billing Efficiency %", "Billing Eff %"), use_container_width=True)

    with tabs[4]: # Human Resource
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Employee Count", "Total Employee Count"), use_container_width=True)
        with c2:
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Bar(x=df_view['Period'], y=df_view['New Hires'], name='New Hires', marker_color='#2ecc71'))
            fig_hr.add_trace(go.Bar(x=df_view['Period'], y=-df_view['Departures'], name='Departures', marker_color='#e74c3c'))
            fig_hr.update_layout(barmode='relative', title="HR Flow (Inflow/Outflow)", template="plotly_white")
            st.plotly_chart(fig_hr, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Recruitment / Rev %", "Recruitment Spend % of Revenue"), use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Recruitment / Rev per CG %", "Recruitment Spend / Rev per Caregiver %"), use_container_width=True)

    with tabs[5]: # Clients
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Client Count", "Total Active Clients"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Avg Contract Duration", "Avg Contract Duration (Days)"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig_cl = go.Figure()
            fig_cl.add_trace(go.Bar(x=df_view['Period'], y=df_view['New Clients'], name='New Clients', marker_color='#2ecc71'))
            fig_cl.add_trace(go.Bar(x=df_view['Period'], y=-df_view['Lost Clients'], name='Lost Clients', marker_color='#e74c3c'))
            fig_cl.update_layout(barmode='relative', title="Client Flows (Inflow/Outflow)", template="plotly_white")
            st.plotly_chart(fig_cl, use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Contact Hours per Client", "Avg Per Client Contact Hrs"), use_container_width=True)

    with tabs[6]: # Marketing & Sales Metrics
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Marketing Spend", "Total Marketing Spend"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "CAC", "Customer Acquisition Cost (CAC)"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig_cac = go.Figure()
            fig_cac.add_trace(go.Scatter(x=df_view['Period'], y=df_view['CAC / Rev per Client %'], name='CAC / Rev per Client %'))
            fig_cac.add_trace(go.Scatter(x=df_view['Period'], y=df_view['CAC / COGS per Client %'], name='CAC / COGS per Client %'))
            fig_cac.update_layout(title="CAC Efficiency %", template="plotly_white")
            st.plotly_chart(fig_cac, use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Sales Conv %", "Avg Sales Conversion %"), use_container_width=True)

    with tabs[7]: # Raw Data
        st.dataframe(df_view.style.format({"Revenue": "${:,.0f}", "EBITDA": "${:,.0f}"}), use_container_width=True)

# ==========================================
# 6. VIEW: FINANCIAL PROJECTIONS
# ==========================================
def show_projections():
    st.title("🚀 Financial Projections")

    st.subheader("Step 1: Configuration & Baseline")
    st.info("Choose the baseline configuration to build your financial projections assumptions.")
    
    c1, c2, c3, c4, c5 = st.columns(5) 
    with c1:
        st.markdown("<div class='step-header'>Entity:</div>", unsafe_allow_html=True)
        target_sub = st.selectbox("", ["Magna Vita (Consolidated)"] + list(df_master["Subsidiary"].unique()))
    with c2:
        st.markdown("<div class='step-header'>Baseline Period:</div>", unsafe_allow_html=True)
        hist_basis = st.selectbox("", ["Weekly", "Monthly", "Quarterly", "Annual"])
    with c3:
        st.markdown("<div class='step-header'>Baseline Year:</div>", unsafe_allow_html=True)
        avail_years = [2025, 2024]
        basis_year = st.selectbox("", avail_years, index=0)
    with c4:
        st.markdown("<div class='step-header'>Projections Duration:</div>", unsafe_allow_html=True)
        proj_years = st.radio("", ["1 Quarter", "1 Year", "3 Years", "5 Years"], horizontal=True)
    with c5:
        st.markdown("<div class='step-header'>Service Type:</div>", unsafe_allow_html=True)
        target_svc = st.multiselect("", options=sorted(df_master["Service Type"].unique()), default=df_master["Service Type"].unique())

    hist_cli = get_historical_avg(df_master, target_sub, "Client Count", freq=hist_basis, year=basis_year, service_types=target_svc)
    
    st.divider()

    st.subheader("Step 2: Enter Baseline Assumptions")
    st.info(f"Enter the starting assumptions based on {basis_year} ({hist_basis}) data for selected services.")

    inputs = {}
    st.markdown("#### 1. Revenue Related Assumptions")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        inputs['n_patients'] = st.number_input(f"Existing Patients (Preceding Period)", value=int(hist_cli) if hist_cli > 0 else 80)
        inputs['new_vs_existing_pct'] = st.number_input(f"New / Existing Patient %", value=5.0) / 100
    with r2:
        inputs['sched_qual_pct'] = st.number_input(f"Sched/Qualified Hrs % Per Patient", value=95.0) / 100
        inputs['time_served_pct'] = st.number_input(f"% Time Served in Period", value=100.0) / 100
    with r3:
        inputs['contact_sched_pct'] = st.number_input(f"Contact/Sched Hrs % Per Patient", value=90.0) / 100
        inputs['service_rate'] = st.number_input("Service Reimbursement Rate ($/hr)", value=32.0)
    with r4:
        inputs['bill_contact_pct'] = st.number_input(f"Billable/Contact Hrs % Per Patient", value=98.0) / 100
        default_hrs = 160 if hist_basis == "Monthly" else (480 if hist_basis == "Quarterly" else 1920)
        if hist_basis == "Weekly": default_hrs = 40
        inputs['base_qual_hours'] = st.number_input(f"Base Qualified Hrs Per Patient", value=default_hrs)

    st.markdown("---")
    st.markdown("#### 2. COGS Related Assumptions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inputs['wage_unskilled'] = st.number_input("Avg Unskilled Caregiver Rate ($/hr)", value=17.0)
        inputs['wage_skilled'] = st.number_input("Avg Skilled Caregiver Rate ($/hr)", value=28.0)
    with c2:
        inputs['benefit_pct_unskilled'] = st.number_input("Benefit % (Unskilled FT)", value=15.0) / 100
        inputs['benefit_pct_skilled'] = st.number_input("Benefit % (Skilled FT)", value=20.0) / 100
    with c3:
        default_unskilled = int(hist_cli * 0.4) if hist_cli > 0 else 30
        default_skilled = int(hist_cli * 0.2) if hist_cli > 0 else 15
        inputs['n_unskilled'] = st.number_input("Total Number of Unskilled Caregivers", value=default_unskilled)
        inputs['n_skilled'] = st.number_input("Total Number of Skilled Caregivers", value=default_skilled)
    with c4:
        inputs['tax_pct_labor_unskilled'] = st.number_input("Payroll Tax % Unskilled", value=12.0) / 100
        inputs['tax_pct_labor_skilled'] = st.number_input("Payroll Tax % Skilled", value=12.0) / 100
        total_cg = inputs['n_unskilled'] + inputs['n_skilled']
        inputs['skilled_mix_pct'] = inputs['n_skilled'] / total_cg if total_cg > 0 else 0.3

    st.markdown("---")
    st.markdown("#### 3. Expenses Related Assumptions")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1: inputs['sal_office'] = st.number_input(f"Office Staff Salaries ({hist_basis})", value=15000.0)
    with e2: inputs['cost_hiring'] = st.number_input(f"Hiring Costs ({hist_basis})", value=2000.0)
    with e3: inputs['cost_marketing'] = st.number_input(f"Marketing Costs ({hist_basis})", value=3000.0)
    with e4: inputs['cost_sales'] = st.number_input(f"Sales Costs ({hist_basis})", value=1500.0)
    with e5: inputs['cost_mgmt'] = st.number_input(f"Office Mgmt Costs ({hist_basis})", value=5000.0)

    st.markdown("---")
    st.markdown("#### 4. Other Assumptions")
    o1, o2, o3 = st.columns(3)
    with o1: inputs['tax_rate_corp'] = st.number_input("Avg Tax Rate %", value=25.0) / 100
    with o2: inputs['da_exp'] = st.number_input(f"Depreciation & Amortization ({hist_basis})", value=1000.0)
    with o3: inputs['other_exp'] = st.number_input(f"Other Expenses ({hist_basis})", value=500.0)

    st.divider()
    st.subheader("Step 3: Define Drivers (Annual % Change)")
    st.info("Enter the expected annual % change for each driver.")

    scenarios = ["Robust", "Base", "Conservative"]
    drivers = {}
    cols = st.columns(3)
    defaults = {
        "Robust": {"pat": 10.0, "rate": 3.0, "wage": 2.0, "exp": 2.0},
        "Base": {"pat": 5.0, "rate": 1.0, "wage": 3.0, "exp": 3.0},
        "Conservative": {"pat": 0.0, "rate": 0.0, "wage": 5.0, "exp": 5.0}
    }

    for idx, sc in enumerate(scenarios):
        with cols[idx]:
            st.markdown(f"### {sc}")
            with st.expander("Revenue Drivers", expanded=True):
                drivers[f"{sc}_pat_growth"] = st.number_input(f"Patient Growth %", value=defaults[sc]["pat"], key=f"{sc}_pat") / 100
                drivers[f"{sc}_rate_growth"] = st.number_input(f"Reimb. Rate Growth %", value=defaults[sc]["rate"], key=f"{sc}_rate") / 100
            with st.expander("COGS Drivers", expanded=False):
                drivers[f"{sc}_wage_growth"] = st.number_input(f"Wage Growth %", value=defaults[sc]["wage"], key=f"{sc}_wage") / 100
            with st.expander("Expense Drivers", expanded=False):
                drivers[f"{sc}_exp_growth"] = st.number_input(f"Gen. Expense Growth %", value=defaults[sc]["exp"], key=f"{sc}_exp") / 100

    if hist_basis == "Monthly": steps_per_year = 12
    elif hist_basis == "Quarterly": steps_per_year = 4
    elif hist_basis == "Weekly": steps_per_year = 52
    else: steps_per_year = 1
        
    duration_map = {"1 Quarter": 0.25, "1 Year": 1, "3 Years": 3, "5 Years": 5}
    total_steps = int(duration_map[proj_years] * steps_per_year)
    if total_steps < 1: total_steps = 1

    # Valuation Multiples
    multiples_map = {
        "All About Caring - Rush City": {"Conservative": 4.5, "Base": 6.0, "Robust": 7.5},
        "All About Caring - Twin Cities": {"Conservative": 4.5, "Base": 6.0, "Robust": 7.5},
        "Communities of Care": {"Conservative": 6.0, "Base": 8.0, "Robust": 10.0},
        "Magna Vita (Consolidated)": {"Conservative": 5.0, "Base": 6.6, "Robust": 8.3}
    }

    def calculate_projection(sc):
        pat_g = drivers[f"{sc}_pat_growth"]
        rate_g = drivers[f"{sc}_rate_growth"]
        wage_g = drivers[f"{sc}_wage_growth"]
        exp_g = drivers[f"{sc}_exp_growth"]

        p_pat_g = (1 + pat_g) ** (1/steps_per_year) - 1
        p_rate_g = (1 + rate_g) ** (1/steps_per_year) - 1
        p_wage_g = (1 + wage_g) ** (1/steps_per_year) - 1
        p_exp_g = (1 + exp_g) ** (1/steps_per_year) - 1

        curr_patients = inputs['n_patients'] * (1 + inputs['new_vs_existing_pct'])
        curr_rate = inputs['service_rate']
        curr_wage_u = inputs['wage_unskilled']
        curr_wage_s = inputs['wage_skilled']
        curr_opex_base = (inputs['sal_office'] + inputs['cost_hiring'] + inputs['cost_marketing'] + inputs['cost_sales'] + inputs['cost_mgmt'] + inputs['other_exp'])

        data_rows = []
        start_date = date.today()
        
        mult = multiples_map.get(target_sub, multiples_map["Magna Vita (Consolidated)"])[sc]

        for i in range(1, total_steps + 1):
            curr_patients *= (1 + p_pat_g)
            curr_rate *= (1 + p_rate_g)
            curr_wage_u *= (1 + p_wage_g)
            curr_wage_s *= (1 + p_wage_g)
            curr_opex_base *= (1 + p_exp_g)

            hrs_qual = inputs['base_qual_hours']
            hrs_sched = inputs['base_qual_hours'] * inputs['sched_qual_pct']
            hrs_contact = hrs_sched * inputs['contact_sched_pct']
            hrs_bill = hrs_contact * inputs['bill_contact_pct']
            
            total_billable_hours = curr_patients * hrs_bill * inputs['time_served_pct']
            revenue = total_billable_hours * curr_rate

            total_labor_hours = curr_patients * hrs_contact
            cost_u = curr_wage_u * (1 + inputs['tax_pct_labor_unskilled'] + inputs['benefit_pct_unskilled'])
            cost_s = curr_wage_s * (1 + inputs['tax_pct_labor_skilled'] + inputs['benefit_pct_skilled'])
            avg_hourly_cost = (cost_s * inputs['skilled_mix_pct']) + (cost_u * (1 - inputs['skilled_mix_pct']))
            cogs = total_labor_hours * avg_hourly_cost

            gross_margin = revenue - cogs
            ebitda = gross_margin - curr_opex_base
            net_income = (ebitda - inputs['da_exp']) * (1 - inputs['tax_rate_corp'])
            
            valuation = ebitda * mult

            if hist_basis == "Monthly":
                date_val = pd.to_datetime(start_date) + pd.DateOffset(months=i)
                lbl = f"{date_val.year}-{date_val.month:02d}"
            elif hist_basis == "Quarterly":
                date_val = pd.to_datetime(start_date) + pd.DateOffset(months=i*3)
                lbl = f"{date_val.year}-Q{(date_val.month-1)//3+1}"
            elif hist_basis == "Weekly":
                date_val = pd.to_datetime(start_date) + pd.DateOffset(weeks=i)
                lbl = f"{date_val.year}-W{date_val.isocalendar().week:02d}"
            else:
                date_val = pd.to_datetime(start_date) + pd.DateOffset(years=i)
                lbl = str(date_val.year)

            data_rows.append({
                "Scenario": sc, "Period": lbl, 
                "Revenue": revenue, "EBITDA": ebitda, "Net Income": net_income,
                "Valuation": valuation,
                "Gross Margin %": (gross_margin/revenue)*100 if revenue else 0,
                "EBITDA Margin %": (ebitda/revenue)*100 if revenue else 0,
                "Net Margin %": (net_income/revenue)*100 if revenue else 0
            })
        return pd.DataFrame(data_rows)

    df_all = pd.concat([calculate_projection(sc) for sc in scenarios])

    st.divider()
    st.subheader(f"📊 Projected Performance Visuals")
    
    tab_vis, tab_data = st.tabs(["Charts", "Data Table"])
    with tab_vis:
        st.markdown("#### Financial Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_rev = px.line(df_all, x="Period", y="Revenue", color="Scenario", markers=True, title="Projected Revenue")
            st.plotly_chart(fig_rev, use_container_width=True)
        with c2:
            fig_ebitda = px.line(df_all, x="Period", y="EBITDA", color="Scenario", markers=True, title="Projected EBITDA")
            st.plotly_chart(fig_ebitda, use_container_width=True)
        with c3:
            fig_ni = px.line(df_all, x="Period", y="Net Income", color="Scenario", markers=True, title="Projected Net Income")
            st.plotly_chart(fig_ni, use_container_width=True)
            
        st.divider()
        st.markdown("#### Margin Analysis")
        c4, c5, c6 = st.columns(3)
        with c4:
            fig_gm = px.line(df_all, x="Period", y="Gross Margin %", color="Scenario", markers=True, title="Gross Margin %")
            st.plotly_chart(fig_gm, use_container_width=True)
        with c5:
            fig_em = px.line(df_all, x="Period", y="EBITDA Margin %", color="Scenario", markers=True, title="EBITDA Margin %")
            st.plotly_chart(fig_em, use_container_width=True)
        with c6:
            fig_nm = px.line(df_all, x="Period", y="Net Margin %", color="Scenario", markers=True, title="Net Margin %")
            st.plotly_chart(fig_nm, use_container_width=True)
            
        st.divider()
        st.markdown("#### Subsidiary Valuation")
        fig_val = px.bar(df_all, x="Period", y="Valuation", color="Scenario", barmode="group",
                         title=f"Projected Valuation (Based on Market EBITDA Multiples for {target_sub})")
        fig_val.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_val, use_container_width=True)
        st.info("ℹ️ Valuation calculated using EBITDA multiples: Conservative (~4.5x-6x), Base (~6x-8x), Robust (~7.5x-10x) depending on subsidiary type.")
        csv_proj = df_all.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Projections as CSV",
            data=csv_proj,
            file_name=f"MV_Projections_{date.today()}.csv",
            mime="text/csv",
        )
        
        with tab_data:
            st.subheader("Raw Projection Data")
            st.dataframe(df_all, use_container_width=True)

    with tab_data:
        pivot = df_all.pivot(index="Period", columns="Scenario", values=["Revenue", "EBITDA", "Net Income", "Valuation"])
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)
        
# ==============================================================================
# 7. VIEW: ACQUISITION SIMULATION (FINAL INTEGRATED VERSION)
# ==============================================================================
def show_acquisition():
    # --- 0. INITIALIZE ALL VARIABLES (Prevents NameErrors & Fixes Resets) ---
    t_rev = 0.0
    t_ebitda = 0.0
    ebitda_mult = 0.0
    retention_rate = 0.0
    organic_growth = 0.0
    synergy_pct = 0.0
    fcf_conv = 0.0
    debt_pct = 0.0
    consolidated_ebitda = 0.0
    total_debt_service = 0.0
    adj_t_ebitda = 0.0
    mv_ebitda_base = 0.0
    s_t_ebitda = 0.0
    adj_t_ebitda = 0.0
    synergy_pct = 0.0
    organic_growth = 0.0

    target_max_lev_hurdle = None
    target_min_ebitda_hurdle = None
    target_max_price_hurdle = None
    target_available_cash = None
    target_min_dscr_hurdle = None
    roi_input_val = None
    target_min_roi_hurdle = 0.0
    
    apply_stress = False
    stress_retention_drop = 0.0
    stress_rate_jump = 0.0

    # --- 1. SESSION STATE & RESET LOGIC ---
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    if "del_rc" not in st.session_state:
        st.session_state.del_rc = 0
        
    rc = st.session_state.reset_counter
    drc = st.session_state.del_rc
    
    db_file = "acquisition_history.csv"
    db_cols = ["Sim_ID", "Timestamp", "Simulator", "Target", "State", "City", "Verdict", "ROI", "Leverage", "Reasons"]
    if os.path.isfile(db_file):
        df_hist_master = pd.read_csv(db_file)
        df_hist_master["Timestamp_DT"] = pd.to_datetime(df_hist_master["Timestamp"], errors='coerce')
    else:
        df_hist_master = pd.DataFrame(columns=db_cols)

    st.title("🤝 Strategic Acquisition Simulation")

    # --- 2. INPUT BLOCKS ---
    st.markdown("<div class='section-header-blue'>👤 Simulator Identity</div>", unsafe_allow_html=True)
    id1, id2, id3 = st.columns(3)
    with id1: sim_first = st.text_input("First Name", value="", key=f"sf_{rc}")
    with id2: sim_last = st.text_input("Last Name", value="", key=f"sl_{rc}")
    with id3: sim_pos = st.selectbox("Position", ["", "CEO", "Director", "Analyst", "Consultant"], key=f"sp_{rc}")
    full_sim_name = f"{sim_first} {sim_last}".strip()

    st.markdown("<div class='section-header-blue'>⚖️ Investment Mandate</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1: target_max_lev_hurdle = st.number_input("Max Leverage Hurdle", 1.0, 10.0, value=None, key=f"h1_{rc}")
    with p2: target_min_ebitda_hurdle = st.number_input("Min EBITDA Hurdle ($)", 0, 100000000, value=None, key=f"h2_{rc}")
    with p3: target_max_price_hurdle = st.number_input("Max Multiple Hurdle", 1.0, 25.0, value=None, key=f"h3_{rc}")
    with p4: target_available_cash = st.number_input("Available Cash ($)", 0, 500000000, value=None, key=f"h4_{rc}")

    p5, p6, p7 = st.columns(3)
    with p5: target_min_dscr_hurdle = st.number_input("Min DSCR Hurdle", 0.0, 5.0, value=None, step=0.1, key=f"h5_{rc}")
    with p6: 
        roi_input_val = st.number_input("Min Year 1 ROI (%)", 0.0, 100.0, value=None, step=1.0, key=f"h6_{rc}")
        target_min_roi_hurdle = (roi_input_val / 100) if roi_input_val else 0.0
    with p7: check_accretion = st.toggle("Hurdle: Margin Accretive", value=False, key=f"h7_{rc}")

    st.markdown("<div class='section-header-blue'>Step 1: Target Basic Information</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: target_name = st.text_input("Target Name", value="", key=f"tn_{rc}")
    with c2: target_state = st.selectbox("State", ["", "Minnesota", "Wisconsin", "Iowa", "Illinois", "Florida"], index=0, key=f"ts_{rc}")
    with c3: target_city = st.text_input("City", value="", key=f"tc_{rc}")
    with c4: target_closing = st.date_input("Estimated Closing Date", value=None, key=f"td_{rc}")

    st.markdown("<div class='section-header-blue'>Step 2: Financial Assumptions</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: 
        t_rev = st.number_input("Target TTM Revenue ($)", value=None, key=f"f1a_{rc}")
        t_ebitda = st.number_input("Target TTM EBITDA ($)", value=None, key=f"f1b_{rc}")
    with f2:
        retention_rate = st.slider("Retention (%)", 0, 100, 0, key=f"f2a_{rc}") / 100
        organic_growth = st.slider("Year 1 Growth (%)", -50, 100, 0, key=f"f2b_{rc}") / 100
    with f3:
        ebitda_mult = st.slider("Purchase Multiple (x)", 0.0, 20.0, 0.0, 0.1, key=f"f3a_{rc}")
        synergy_pct = st.slider("Synergy Potential (%)", -50, 100, 0, key=f"f3b_{rc}") / 100
    with f4:
        fcf_conv = st.slider("FCF Conversion (%)", 0, 100, 0, key=f"f4a_{rc}") / 100
        s_t_ebitda = t_ebitda if t_ebitda else 0
        purchase_price = s_t_ebitda * ebitda_mult
        st.metric("Implied Purchase Price", f"${purchase_price:,.0f}")

    st.markdown("<div class='section-header-blue'>Step 3: Financing</div>", unsafe_allow_html=True)

    # 1. Input Layout: Percentage Allocation and Terms
    fi1, fi2, fi3 = st.columns([1, 1.5, 1.5])

    with fi1:
        st.markdown("**Allocation**")
        debt_pct = st.slider("Bank Loan (%)", 0, 100, 0, key=f"fi1a_{rc}") / 100
        seller_pct = st.slider("Seller Note (%)", 0, 100, 0, key=f"fi1b_{rc}") / 100
        transaction_fees = st.number_input("Est. Transaction Fees ($)", value=None, key=f"fi2c_{rc}")

    with fi2:
        st.markdown("**Bank Loan Terms**")
        bank_int_input = st.number_input("Bank Interest Rate (%)", value=None, key=f"bank_int_{rc}")
        bank_loan_term = st.number_input("Bank Term (Years)", value=None, step=1, key=f"bank_term_{rc}")

    with fi3:
        st.markdown("**Seller Note Terms**")
        seller_int_input = st.number_input("Seller Interest (%)", value=None, key=f"sell_int_{rc}")
        seller_loan_term = st.number_input("Seller Term (Years)", value=None, step=1, key=f"sell_term_{rc}")
    
    # --- ADD ZERO-GUARDING HERE ---
    b_term = bank_loan_term if (bank_loan_term and bank_loan_term > 0) else 1
    s_term = seller_loan_term if (seller_loan_term and seller_loan_term > 0) else 1

    # 2. Sanitized Math & Split Debt Service
    b_rate = (bank_int_input / 100) if bank_int_input else 0.0
    s_rate = (seller_int_input / 100) if seller_int_input else 0.0
    s_fees = transaction_fees if transaction_fees else 0
    
    # ALIAS for legacy code: ensures 'int_rate_val' exists for your heatmaps
    int_rate_val = b_rate 
    
    s_new_debt = purchase_price * debt_pct
    s_seller_note = purchase_price * seller_pct
    total_financing = s_new_debt + s_seller_note
    cash_equity_needed = (purchase_price + s_fees) - total_financing
    
    # --- STEP 1: FINANCIAL PROJECTIONS (The "Numerator") ---
    adj_t_ebitda = s_t_ebitda * (calc_ret if apply_stress else 1.0)
    
    consolidated_ebitda = (mv_ebitda_base + adj_t_ebitda) + (adj_t_ebitda * synergy_pct)
    consolidated_ebitda += (consolidated_ebitda * organic_growth)

    # --- STEP 2: DEBT RATE DEFINITIONS ---
    base_b_rate = b_rate  
    base_s_rate = s_rate

    # DYNAMICALLY DETECT THE STRESS VARIABLE
    # This looks for 'interest_stress' or 'st_int_adj' or 'stress_rate_jump'
    if 'interest_stress' in locals():
        stress_rate_jump = interest_stress
    elif 'st_int_adj' in locals():
        stress_rate_jump = st_int_adj
    else:
        stress_rate_jump = 0.0

    # Apply stress
    stressed_b_rate = base_b_rate + (stress_rate_jump if apply_stress else 0)
    stressed_s_rate = base_s_rate + (stress_rate_jump if apply_stress else 0)
    
    # ALIAS for legacy code: ensures 'calc_int' is updated for ROI heatmaps
    calc_int = stressed_b_rate

    # --- STEP 3: DEBT SERVICE MATH ---
    bank_annual_pri = s_new_debt / b_term
    seller_annual_pri = s_seller_note / s_term
    
    base_debt_service = (bank_annual_pri + (s_new_debt * base_b_rate)) + \
                        (seller_annual_pri + (s_seller_note * base_s_rate))

    stressed_debt_service = (bank_annual_pri + (s_new_debt * stressed_b_rate)) + \
                            (seller_annual_pri + (s_seller_note * stressed_s_rate))

    # --- STEP 4: KPI MATH WITH DELTAS ---
    # Base Case (Record Vault)
    deal_dscr = (consolidated_ebitda * fcf_conv) / base_debt_service if base_debt_service > 0 else 0
    base_roi = ((mv_ebitda_base + s_t_ebitda) * fcf_conv - base_debt_service) / cash_equity_needed * 100 if cash_equity_needed > 0 else 0
    
    # Live Case (Dashboard UI)
    live_dscr = (consolidated_ebitda * fcf_conv) / stressed_debt_service if stressed_debt_service > 0 else 0
    cash_roi = (consolidated_ebitda * fcf_conv - stressed_debt_service) / cash_equity_needed * 100 if cash_equity_needed > 0 else 0

    # Delta Calculations
    dscr_delta_val = live_dscr - deal_dscr
    dscr_delta_pct = (dscr_delta_val / deal_dscr * 100) if deal_dscr > 0 else 0

    ebitda_delta_val = consolidated_ebitda - (mv_ebitda_base + s_t_ebitda)
    ebitda_delta_pct = (ebitda_delta_val / (mv_ebitda_base + s_t_ebitda) * 100) if (mv_ebitda_base + s_t_ebitda) > 0 else 0

    roi_delta_val = cash_roi - base_roi
    roi_delta_pct = (roi_delta_val / base_roi * 100) if base_roi > 0 else 0

    # 3. Summary Metrics Row
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Bank Loan Amt", f"${s_new_debt:,.0f}")
    with m2: st.metric("Seller Note Amt", f"${s_seller_note:,.0f}")
    with m3: st.metric("Total Financing", f"${total_financing:,.0f}")
    with m4: st.metric("Cash Equity Needed", f"${cash_equity_needed:,.0f}", delta_color="inverse")

    # 4. Updated Readiness Checklist
    checklist = {
        "Identity": full_sim_name != "",
        "Mandates": all(v is not None for v in [target_max_lev_hurdle, target_min_ebitda_hurdle, target_max_price_hurdle]),
        "Target": target_name != "" and target_state != "",
        "Financials": s_t_ebitda > 0 and ebitda_mult > 0,
        "Financing Structure": (debt_pct > 0 or seller_pct > 0),
        "Bank Terms": bank_int_input is not None and bank_loan_term is not None,
        "Seller Terms": seller_int_input is not None and seller_loan_term is not None,
        "Closing Costs": transaction_fees is not None
    }

    progress_pct = sum(checklist.values()) / len(checklist)
    bar_color = "#FF4B4B" if progress_pct < 0.4 else "#FFA500" if progress_pct < 1 else "#27AE60"

    st.markdown(f"""
        <div style="width: 100%; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
            <div style="width: {int(progress_pct*100)}%; background-color: {bar_color}; height: 10px; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
        </div>
        <p style="text-align: right; color: {bar_color}; font-weight: bold;">Completion: {int(progress_pct*100)}%</p>
    """, unsafe_allow_html=True)

    if progress_pct < 1.0:
        st.info("💡 Complete required fields to activate Run Simulation.")
        st.button("🚀 Run Simulation", disabled=True, key="dis_btn")
    else:
        if st.button("🚀 Run Simulation", type="primary", key="act_btn"):
            st.session_state[f"run_sim_{rc}"] = True
            st.rerun()

    # --- OUTPUT GATE ---
    if st.session_state.get(f"run_sim_{rc}", False):
        try:
            df_2025 = df_master[df_master['Year'] == 2025]
            mv_rev_base = df_2025['Revenue'].sum()
            mv_ebitda_base = df_2025['EBITDA'].sum()
            mv_fcf_base = mv_ebitda_base * 0.75
            mv_margin_base = (mv_ebitda_base / mv_rev_base * 100) if mv_rev_base > 0 else 0
        except:
            mv_rev_base = mv_ebitda_base = mv_fcf_base = mv_margin_base = 0

        st.markdown("### 🧪 Scenario Stress Testing")
        with st.expander("Configure Shock Parameters", expanded=False):
            c_s1, c_s2 = st.columns(2)
            stress_retention_drop = c_s1.number_input("Retention Shock (% Drop)", 0.0, 100.0, 10.0) / 100
            stress_rate_jump = c_s2.number_input("Interest Rate Shock (bps)", 0.0, 1000.0, 200.0) / 10000
            apply_stress = st.toggle("🚨 Apply Stress Test")
            
            if apply_stress:
                st.warning("⚠️ **Note:** Stress test results are for simulation only. Saving to the Record Vault will archive the **Base Case** (pre-shock) financials.")

        # Engine Math
        calc_ret = retention_rate - (stress_retention_drop if apply_stress else 0)
        calc_int = int_rate_val + (stress_rate_jump if apply_stress else 0)
        adj_t_ebitda = s_t_ebitda * calc_ret
        synergy_val = adj_t_ebitda * synergy_pct
        growth_val = (mv_ebitda_base + adj_t_ebitda) * organic_growth
        consolidated_ebitda = (mv_ebitda_base + adj_t_ebitda) + synergy_val + growth_val
        
        # --- INDEPENDENT DEBT SERVICE CALCULATIONS ---

        # A. Bank Loan Annual Payment
        # Standard P+I calculation: (Principal / Term) + (Principal * Rate)
        bank_annual_pri = s_new_debt / bank_loan_term if (bank_loan_term and bank_loan_term > 0) else 0
        bank_annual_int = s_new_debt * b_rate
        total_bank_svc = bank_annual_pri + bank_annual_int

        # B. Seller Note Annual Payment
        seller_annual_pri = s_seller_note / seller_loan_term if (seller_loan_term and seller_loan_term > 0) else 0
        seller_annual_int = s_seller_note * s_rate
        total_seller_svc = seller_annual_pri + seller_annual_int

        # C. Total Combined Debt Service
        # This is the 'Denominator' for your DSCR
        total_annual_debt_service = total_bank_svc + total_seller_svc

        # --- UPDATED CASH FLOW MATH ---

        # We use the 'adj_t_ebitda' (which handles stress testing if active)
        # and 'fcf_conv' (Free Cash Flow conversion percentage)
        annual_fcf_pre_debt = adj_t_ebitda * fcf_conv

        # Year 1 Cash Flow After All Debt
        annual_fcf_post_debt = annual_fcf_pre_debt - total_annual_debt_service

        # DSCR Calculation
        deal_dscr = (annual_fcf_pre_debt / total_annual_debt_service) if total_annual_debt_service > 0 else 0

        # Cash ROI Calculation
        cash_roi = (annual_fcf_post_debt / cash_equity_needed * 100) if cash_equity_needed > 0 else 0
        
        pf_rev = (mv_rev_base + (t_rev or 0)) * (1 + organic_growth)
        pf_margin = (consolidated_ebitda / pf_rev * 100) if pf_rev > 0 else 0

        # Verdict
        # Calculate Total Leverage (Debt / EBITDA)
        total_lev = (s_new_debt + s_seller_note) / (consolidated_ebitda if consolidated_ebitda > 0 else 1)
        net_fcf = annual_fcf_post_debt
        
        # Define Buffers for Caution (e.g., within 10% of a hurdle)
        dscr_buffer = (target_min_dscr_hurdle or 1.2) * 1.10
        lev_buffer = (target_max_lev_hurdle or 4.0) * 0.90

        fail_reasons = []
        caution_reasons = []

        # -- FAIL LOGIC --
        if total_lev > (target_max_lev_hurdle or 10): fail_reasons.append("Leverage Limit")
        if deal_dscr < (target_min_dscr_hurdle or 0): fail_reasons.append("DSCR Floor")
        if (cash_roi/100) < target_min_roi_hurdle: fail_reasons.append("ROI Goal")

        # -- CAUTION LOGIC (Only check if not already failed) --
        if not fail_reasons:
            if total_lev > lev_buffer: caution_reasons.append("High Leverage (Near Limit)")
            if deal_dscr < dscr_buffer: caution_reasons.append("Tight Debt Coverage")
            if apply_stress: caution_reasons.append("Stressed Scenario Active")

        # Determine Final Verdict
        if fail_reasons:
            verdict, v_color = "FAIL: DE-PRIORITIZE", "#e74c3c" # Red
            sub_text = " | ".join(fail_reasons)
        elif caution_reasons:
            verdict, v_color = "CAUTION: PROCEED WITH CARE", "#f1c40f" # Yellow/Amber
            sub_text = " | ".join(caution_reasons)
        else:
            verdict, v_color = "PASS: STRATEGIC FIT", "#27ae60" # Green
            sub_text = "Investment Mandates Cleared"
        
        # Render Verdict Banner
        text_color = "white" if v_color != "#f1c40f" else "black"
        st.markdown(f"""
            <div style='background-color:{v_color}; padding:20px; border-radius:10px; text-align:center; color:{text_color};'>
                <h2 style='margin:0;'>{verdict}</h2>
                <p style='margin:5px 0 0 0; font-weight:bold;'>{sub_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # 8-GRID KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("DSCR", f"{deal_dscr:.2f}x", delta=f"{deal_dscr - (target_min_dscr_hurdle or 0):.2f} Hurdle")
        k2.metric("Consolidated EBITDA", f"${consolidated_ebitda:,.0f}", delta=f"${consolidated_ebitda - (target_min_ebitda_hurdle or 0):,.0f} Goal")
        k3.metric("Net Free Cash Flow", f"${net_fcf:,.0f}", delta=f"${net_fcf - mv_fcf_base:,.0f} MV Base")
        k4.metric("Cash ROI", f"{cash_roi:.1f}%", delta=f"{cash_roi - (target_min_roi_hurdle*100):.1f}% Goal")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Debt / EBITDA", f"{total_lev:.2f}x", delta=f"{total_lev - (target_max_lev_hurdle or 0):.2f} Limit", delta_color="inverse")
        k6.metric("Margin %", f"{pf_margin:.1f}%", delta=f"{pf_margin - mv_margin_base:.1f}% vs MV Base")
        k7.metric("Cash Required", f"${cash_equity_needed:,.0f}", delta=f"${(target_available_cash or 0) - cash_equity_needed:,.0f} Budget")
        k8.metric("Purchase Multiple", f"{ebitda_mult:.1f}x", delta=f"{ebitda_mult - (target_max_price_hurdle or 0):.1f} Limit", delta_color="inverse")

        # 6. TAB NAVIGATION (Consolidated)
        st.divider()
        tabs = st.tabs(["📊 Financial Bridge", "📈 Amortization", "🧪 Sensitivity Analysis", "📝 Record Management"])

        
        # --- TAB 0: FINANCIAL ANALYSIS ---
        with tabs[0]:
            st.subheader("EBITDA Contribution Waterfall")
            fig_bridge = go.Figure(go.Waterfall(
                orientation = "v",
                measure = ["relative", "relative", "relative", "total"],
                x = ["2025 MV Base", "Acquisition EBITDA", "Synergies/Growth", "Pro-Forma Total"],
                y = [mv_ebitda_base, adj_t_ebitda, synergy_val + growth_val, consolidated_ebitda],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            st.plotly_chart(fig_bridge, use_container_width=True)

        # --- TAB 1: AMORTIZATION SCHEDULES ---
        with tabs[1]:
            st.subheader("📊 Amortization Schedules")

            # --- 1. DYNAMICALLY CAPTURE SLIDER VALUE ---
            # Use the stress adjustment from the stress testing section above
            current_stress = stress_rate_jump if apply_stress else 0.0
            
            # Use the captured stress value
            active_b_rate = base_b_rate + (current_stress if apply_stress else 0)
            active_s_rate = base_s_rate + (current_stress if apply_stress else 0)

            # Visual feedback to confirm the table is reacting
            if apply_stress and current_stress != 0:
                st.warning(f"📈 **Stress Applied:** Rates adjusted by +{current_stress*100:.2f}%")
            else:
                st.info("💡 Showing Base Case Amortization (No Stress Applied)")

            b_term_val = int(bank_loan_term) if bank_loan_term else 0
            s_term_val = int(seller_loan_term) if seller_loan_term else 0
            max_term = max(b_term_val, s_term_val)
        
            if max_term > 0:
                schedule_list = []
                curr_b_bal = s_new_debt
                curr_s_bal = s_seller_note
        
                # Tracking for cumulative totals
                total_b_i = 0
                total_s_i = 0
        
                for y in range(1, max_term + 1):
                    # --- Bank Math ---
                    if y <= b_term_val:
                        b_p = s_new_debt / b_term_val
                        # Uses the Live Stressed Rate defined at the top of this block
                        b_i = curr_b_bal * active_b_rate 
                        b_pay = b_p + b_i
                        curr_b_bal = max(0, curr_b_bal - b_p)
                    else:
                        b_p = b_i = b_pay = curr_b_bal = 0.0
                
                    # --- Seller Math ---
                    if y <= s_term_val:
                        s_p = s_seller_note / s_term_val
                        # Uses the Live Stressed Rate defined at the top of this block
                        s_i = curr_s_bal * active_s_rate
                        s_pay = s_p + s_i
                        curr_s_bal = max(0, curr_s_bal - s_p)
                    else:
                        s_p = s_i = s_pay = curr_s_bal = 0.0

                    total_b_i += b_i
                    total_s_i += s_i

                    schedule_list.append({
                        "Year": f"Year {y}",
                        "Bank Principal": b_p,
                        "Bank Interest": b_i,
                        "Bank Payment": b_pay,
                        "Bank Balance": curr_b_bal,
                        "Seller Principal": s_p,
                        "Seller Interest": s_i,
                        "Seller Payment": s_pay,
                        "Seller Balance": curr_s_bal,
                        "Combined Total": (b_pay + s_pay)
                    })

                # Add the Cumulative Totals row
                summary_row = {
                    "Year": "CUMULATIVE TOTALS",
                    "Bank Principal": s_new_debt,
                    "Bank Interest": total_b_i,
                    "Bank Payment": s_new_debt + total_b_i,
                    "Bank Balance": 0,
                    "Seller Principal": s_seller_note,
                    "Seller Interest": total_s_i,
                    "Seller Payment": s_seller_note + total_s_i,
                    "Seller Balance": 0,
                    "Combined Total": (s_new_debt + total_b_i + s_seller_note + total_s_i)
                }
            
                df_full_amort = pd.concat([pd.DataFrame(schedule_list), pd.DataFrame([summary_row])], ignore_index=True)

                # 2. CSS for Bold Headers & Colors
                st.markdown("""
                    <style>
                        .stDataFrame th {
                            font-size: 15px !important;
                            font-weight: 800 !important;
                            color: black !important;
                            text-align: center !important;
                        }
                        .stDataFrame th:nth-child(2), .stDataFrame th:nth-child(3), 
                        .stDataFrame th:nth-child(4), .stDataFrame th:nth-child(5) { background-color: #b3e5fc !important; }
                        .stDataFrame th:nth-child(6), .stDataFrame th:nth-child(7), 
                        .stDataFrame th:nth-child(8), .stDataFrame th:nth-child(9) { background-color: #ffe0b2 !important; }
                        .stDataFrame th:nth-child(10) { background-color: #c8e6c9 !important; }
                    </style>
                """, unsafe_allow_html=True)

                # 3. Data Styling
                styled_df = df_full_amort.style.format("${:,.0f}", subset=df_full_amort.columns[1:]) \
                    .set_properties(subset=["Bank Principal", "Bank Interest", "Bank Payment", "Bank Balance"], 
                                   **{'background-color': '#e1f5fe', 'color': 'black'}) \
                    .set_properties(subset=["Seller Principal", "Seller Interest", "Seller Payment", "Seller Balance"], 
                                   **{'background-color': '#fff3e0', 'color': 'black'}) \
                    .set_properties(subset=["Combined Total"], 
                                   **{'font-weight': 'bold', 'background-color': '#f1f8e9'}) \
                    .apply(lambda x: ['font-weight: bold; background-color: #eeeeee' if x.Year == 'CUMULATIVE TOTALS' else '' for _ in x], axis=1)

                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # --- 4. EXCEL EXPORT ---
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_full_amort.to_excel(writer, sheet_name='Amortization_Schedule', index=False)
                    workbook = writer.book
                    worksheet = writer.sheets['Amortization_Schedule']
                
                    b_head = workbook.add_format({'bold': True, 'bg_color': '#b3e5fc', 'border': 1, 'align': 'center'})
                    s_head = workbook.add_format({'bold': True, 'bg_color': '#ffe0b2', 'border': 1, 'align': 'center'})
                    t_head = workbook.add_format({'bold': True, 'bg_color': '#c8e6c9', 'border': 1, 'align': 'center'})
                    sum_fmt = workbook.add_format({'bold': True, 'bg_color': '#eeeeee', 'num_format': '$#,##0', 'border': 1})
                    num_fmt = workbook.add_format({'num_format': '$#,##0'})

                    worksheet.write(0, 0, "Year")
                    for col in range(1, 5): worksheet.write(0, col, df_full_amort.columns[col], b_head)
                    for col in range(5, 9): worksheet.write(0, col, df_full_amort.columns[col], s_head)
                    worksheet.write(0, 9, "Combined Total", t_head)
                
                    last_row = len(df_full_amort)
                    worksheet.set_row(last_row, 20, sum_fmt)
                    worksheet.set_column('A:A', 18)
                    worksheet.set_column('B:J', 16, num_fmt)

                st.download_button(
                    label="📗 Download Detailed Excel Schedule with Totals",
                    data=buffer.getvalue(), 
                    file_name=f"{target_name}_Detailed_Amortization.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("Please enter financing terms to see the schedule.")

        with tabs[2]:
            st.subheader("ROI Risk Analysis")
            
            # --- 1. SET CENTER POINTS ---
            # Use the 'stressed' rate so the middle of the heatmap matches your current dashboard state
            center_rate = stressed_b_rate 
            center_mult = ebitda_mult
            
            # Build sensitivity ranges (+/- 20% from current center)
            m_rng = [center_mult * (1 + x) for x in [-0.2, -0.1, 0, 0.1, 0.2]]
            r_rng = [center_rate * (1 + x) for x in [-0.2, -0.1, 0, 0.1, 0.2]]
            
            sens_data = []
            for m in m_rng:
                row = []
                for r in r_rng:
                    # Logic: If we paid 'm' multiple at 'r' interest rate...
                    tmp_p = s_t_ebitda * m # Purchase Price
                    tmp_d = tmp_p * debt_pct # Bank Debt
                    tmp_s = tmp_p * seller_pct # Seller Note
                    
                    # Calculate Debt Service (Stressed)
                    bank_ds = (tmp_d / (bank_loan_term if bank_loan_term > 0 else 1)) + (tmp_d * r)
                    seller_ds = (tmp_s / (seller_loan_term if seller_loan_term > 0 else 1)) + (tmp_s * stressed_s_rate)
                    
                    # Cash Flow (using the Stressed EBITDA from Step 1 of the engine)
                    tmp_fcf = (consolidated_ebitda * fcf_conv) - (bank_ds + seller_ds)
                    
                    # Cash Equity Required
                    tmp_eq = (tmp_p + s_fees) - (tmp_d + tmp_s)
                    
                    row.append((tmp_fcf / tmp_eq * 100) if tmp_eq > 0 else 0)
                sens_data.append(row)

            df_sens = pd.DataFrame(sens_data, index=[f"{m:.1f}x" for m in m_rng], columns=[f"{r*100:.1f}%" for r in r_rng])
            st.write("### Heatmap: Year 1 Cash ROI %")
            st.info("Center value reflects current Purchase Multiple and Stressed Interest Rate.")
            st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.1f}%"), use_container_width=True)

            st.divider()
            
            # --- 2. LENDING RISK HEATMAP ---
            lev_data = []
            for m in m_rng:
                row = []
                for r in r_rng:
                    tmp_p = s_t_ebitda * m
                    tmp_d = tmp_p * debt_pct
                    tmp_s = tmp_p * seller_pct
                    # Total Leverage Ratio
                    tmp_lev = (tmp_d + tmp_s) / (consolidated_ebitda if consolidated_ebitda > 0 else 1)
                    row.append(tmp_lev)
                lev_data.append(row)

            df_lev = pd.DataFrame(lev_data, index=[f"{m:.1f}x" for m in m_rng], columns=[f"{r*100:.1f}%" for r in r_rng])
            st.write("### Matrix B: Pro-Forma Leverage (Total Debt/EBITDA)")
            st.dataframe(df_lev.style.background_gradient(cmap='Reds_r').format("{:.2f}x"), use_container_width=True)

        with tabs[3]:
            st.subheader("Simulation Record Vault Management")
            
            # --- Initialize a Deletion Reset Counter in Session State if not present ---
            if "del_rc" not in st.session_state:
                st.session_state.del_rc = 0
            drc = st.session_state.del_rc

            # --- 1. FILTER SECTION ---
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                f_name = st.text_input("Search Target Name", key="f_tn")
                f_sim = st.text_input("Search Simulator Name", key="f_sn")
            with c_f2:
                f_city = st.text_input("Search City", key="f_city")
                f_v = st.multiselect("Filter by Verdict", options=df_hist_master["Verdict"].unique() if not df_hist_master.empty else [])
            with c_f3:
                f_state = st.multiselect("Filter by State", options=df_hist_master["State"].unique() if not df_hist_master.empty else [])
                min_ts = df_hist_master["Timestamp_DT"].min().date() if not df_hist_master.empty else date.today()
                f_date = st.date_input("Filter Date Range", value=(min_ts, date.today()))
            
            # --- 2. APPLY FILTERS ---
            mask = pd.Series([True] * len(df_hist_master))
            if f_name: mask &= df_hist_master["Target"].str.contains(f_name, case=False)
            if f_sim: mask &= df_hist_master["Simulator"].str.contains(f_sim, case=False)
            if f_city: mask &= df_hist_master["City"].str.contains(f_city, case=False)
            if f_v: mask &= df_hist_master["Verdict"].isin(f_v)
            if f_state: mask &= df_hist_master["State"].isin(f_state)
            if isinstance(f_date, tuple) and len(f_date) == 2: 
                mask &= (df_hist_master["Timestamp_DT"].dt.date >= f_date[0]) & (df_hist_master["Timestamp_DT"].dt.date <= f_date[1])
            
            v_df = df_hist_master[mask].drop(columns=["Timestamp_DT"]).set_index("Sim_ID")
            st.dataframe(v_df, use_container_width=True)

            if not v_df.empty:
                sel_id = st.selectbox("Select Sim_ID for Action", v_df.index, key=f"v_select_{rc}")
                
                # Show Historical Notes (Vault Version - Non-Editable)
                hist_notes = df_hist_master[df_hist_master["Sim_ID"] == sel_id]["Reasons"].values[0]
                st.info(f"**Historical Rationale for ID {sel_id}:**\n\n{hist_notes}")

                # -------------------------------
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📑 Recover PDF Summary", key=f"v_recov_{sel_id}_{rc}"):
                        # Fetch the specific row as a Series
                        # .iloc[0] ensures 'r' is a Series (subscriptable), not a float
                        r = df_hist_master[df_hist_master["Sim_ID"] == sel_id].iloc[0]
                        s_and_u_hist = {
                            "sources": [
                                ("Bank Debt", f"${s_new_debt:,.0f}"),
                                ("Seller Note", f"${s_seller_note:,.0f}"),
                                ("Cash Equity", f"${cash_equity_needed:,.0f}"),
                                ("Total Sources", f"${(s_new_debt + s_seller_note + cash_equity_needed):,.0f}")
                            ],
                            "uses": [
                                ("Purchase Price", f"${purchase_price:,.0f}"),
                                ("Transaction Fees", f"${s_fees:,.0f}"),
                                ("Total Uses", f"${(purchase_price + s_fees):,.0f}")
                            ]
                        }

                        # Reconstruct Amortization for the PDF Appendix
                        pdf_amort_hist = []
                        max_term_hist = int(max(bank_loan_term or 0, seller_loan_term or 0))
                        r_bal_h = s_new_debt
                        a_pri_h = s_new_debt / bank_loan_term if bank_loan_term and bank_loan_term > 0 else 0
                        for y in range(1, max_term_hist + 1):
                            i_p_h = r_bal_h * int_rate_val # Use the base interest rate
                            t_p_h = a_pri_h + i_p_h
                            r_bal_h -= a_pri_h
                            pdf_amort_hist.append({
                                "Year": f"Y{y}", "Principal": a_pri_h, "Interest": i_p_h, 
                                "Total Payment": t_p_h, "Remaining Balance": max(0, r_bal_h)
                            })

                        # --- BUILD IDENTICAL PDF ---
                        pdf_b = create_pdf(
                            rec={
                                "Target": r["Target"], 
                                "Verdict": r["Verdict"], 
                                "Reasons": r["Reasons"], 
                                "State": r.get("State", "N/A"), 
                                "City": r.get("City", "N/A")
                            },
                            kpi_data={
                                "Purchase Price": f"${purchase_price:,.0f}",
                                "Purchase Multiple": f"{ebitda_mult:.1f}x",
                                "Closing Leverage": r["Leverage"],
                                "Year 1 Cash ROI": r["ROI"],
                                "DSCR (Avg)": f"{deal_dscr:.2f}x"
                            },
                            sources_uses=s_and_u_hist,
                            amort_schedule=pdf_amort_hist
                        )
                        
                        st.download_button(
                            label="📥 Download Recovered Report",
                            data=pdf_b,
                            file_name=f"Recovered_Analysis_{r['Target']}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"dl_v_pdf_{sel_id}"
                        )
                
                with col_b:
                    st.warning(f"Warning: Deleting ID {sel_id} is permanent.")
                    
                    # --- KEY ROTATION FIX ---
                    # We add 'drc' to the key. When drc increments, this checkbox resets to False.
                    confirm_del = st.checkbox(f"I confirm I want to delete record {sel_id}", key=f"del_conf_{sel_id}_{drc}")
                    
                    if confirm_del:
                        if st.button("🗑️ Execute Permanent Delete", type="primary"):
                            # 1. Remove the record
                            df_hist_master = df_hist_master[df_hist_master["Sim_ID"] != sel_id].reset_index(drop=True)
                            
                            # 2. Re-index Sim_ID sequence
                            df_hist_master["Sim_ID"] = range(1, len(df_hist_master) + 1)
                            
                            # 3. Save to CSV
                            df_hist_master.to_csv(db_file, index=False)
                            
                            # --- 4. RESET THE CHECKBOX STATE ---
                            # Increment the sub-counter so the checkbox key changes on next run
                            st.session_state.del_rc += 1
                            
                            st.success(f"✅ DELETION SUCCESSFUL: Record ID {sel_id} has been permanently removed.")
                            
                            import time
                            time.sleep(1) # Delay so the user sees the green success message
                            st.rerun()

        # ==========================================
        # 7. FINAL ACTIONS (Bottom of Page)
        st.divider()
        
        # Initialize button states in session state
        if "vault_saved" not in st.session_state: st.session_state.vault_saved = False
        if "pdf_exported" not in st.session_state: st.session_state.pdf_exported = False

        # --- THE SINGLE RATIONALE BOX ---
        strat_notes = st.text_area("Enter rationale details for this simulation...", key=f"final_notes_{rc}")
        
        if strat_notes.strip():
            st.success("✅ **Rationale Captured:** Ready for Vault/PDF export.")
        else:
            st.info("ℹ️ **Status:** Enter strategic rationale above to include it in the report.")

        st.divider()
        b1, b2, b3 = st.columns(3)
        
        with b1:
            save_label = "✅ Saved to Vault" if st.session_state.vault_saved else "📁 Save Simulation to Record Vault"
            if st.button(save_label, use_container_width=True, disabled=st.session_state.vault_saved, key=f"save_btn_{rc}"):
                
                # --- BACKGROUND BASE CASE CALCULATION ---
                # We ignore apply_stress and use the original variables
                base_adj_ebitda = s_t_ebitda * retention_rate
                base_synergy = base_adj_ebitda * synergy_pct
                base_growth = (mv_ebitda_base + base_adj_ebitda) * organic_growth
                base_cons_ebitda = (mv_ebitda_base + base_adj_ebitda) + base_synergy + base_growth
                
                base_ann_debt_svc = (s_new_debt / (bank_loan_term if bank_loan_term and bank_loan_term > 0 else 1)) + (s_new_debt * int_rate_val)
                base_fcf = (base_cons_ebitda * fcf_conv) - base_ann_debt_svc
                base_roi = (base_fcf / (cash_equity_needed if cash_equity_needed > 0 else 1)) * 100
                base_lev = (s_new_debt + s_seller_note + 1500000) / (base_cons_ebitda if base_cons_ebitda > 0 else 1)

                new_entry = {
                    "Sim_ID": len(df_hist_master)+1, 
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Simulator": full_sim_name, 
                    "Target": target_name, 
                    "State": target_state, 
                    "City": target_city,
                    "Verdict": verdict, # Keeps the verdict based on current screen view
                    "ROI": f"{base_roi:.1f}%", 
                    "Leverage": f"{base_lev:.2f}x", 
                    "Reasons": f"[BASE CASE] {strat_notes}"
                }
                
                pd.concat([df_hist_master, pd.DataFrame([new_entry])]).to_csv(db_file, index=False)
                st.session_state.vault_saved = True
                st.success("Analysis archived as BASE CASE.")
                st.rerun()

        with b2:
            # --- PHASE 1: PDF NOT YET GENERATED ---
            if not st.session_state.pdf_exported:
                if st.button("📄 Export Summary PDF", use_container_width=True, key=f"export_trigger_{rc}"):
                    # 1. Prepare Financing Structure (Sources & Uses)
                    s_and_u = {
                        "sources": [
                            ("Bank Debt", f"${s_new_debt:,.0f}"),
                            ("Seller Note", f"${s_seller_note:,.0f}"),
                            ("Cash Equity", f"${cash_equity_needed:,.0f}"),
                            ("Total Sources", f"${(s_new_debt + s_seller_note + cash_equity_needed):,.0f}")
                        ],
                        "uses": [
                            ("Purchase Price", f"${purchase_price:,.0f}"),
                            ("Transaction Fees", f"${s_fees:,.0f}"),
                            ("Total Uses", f"${(purchase_price + s_fees):,.0f}")
                        ]
                    }

                    # 2. Prepare Amortization Schedule (FORCING BASE CASE DATA)
                    pdf_amort_combined = []
                    m_term = int(max(bank_loan_term or 0, seller_loan_term or 0))
                    
                    # Reset balances for the internal PDF loop
                    b_bal_pdf = s_new_debt
                    s_bal_pdf = s_seller_note

                    for y in range(1, m_term + 1):
                        # Bank Logic (Using base_b_rate)
                        if y <= (bank_loan_term or 0):
                            b_p = s_new_debt / bank_loan_term
                            b_i = b_bal_pdf * base_b_rate  # <--- USES BASE RATE
                            b_pay = b_p + b_i
                            b_bal_pdf = max(0, b_bal_pdf - b_p)
                        else:
                            b_p = b_i = b_pay = b_bal_pdf = 0.0

                        # Seller Logic (Using base_s_rate)
                        if y <= (seller_loan_term or 0):
                            s_p = s_seller_note / seller_loan_term
                            s_i = s_bal_pdf * base_s_rate  # <--- USES BASE RATE
                            s_pay = s_p + s_i
                            s_bal_pdf = max(0, s_bal_pdf - s_p)
                        else:
                            s_p = s_i = s_pay = s_bal_pdf = 0.0

                        pdf_amort_combined.append({
                            "Year": f"Year {y}",
                            "Bank Principal": b_p,
                            "Bank Interest": b_i,
                            "Bank Payment": b_pay,
                            "Bank Balance": b_bal_pdf,
                            "Seller Principal": s_p,
                            "Seller Interest": s_i,
                            "Seller Payment": s_pay,
                            "Seller Balance": s_bal_pdf,
                            "Combined Total": (b_pay + s_pay)
                        })

                    # 3. Build PDF
                    pdf_b = create_pdf(
                        rec={
                            "Target": target_name, 
                            "Verdict": verdict, 
                            "Reasons": strat_notes, 
                            "State": target_state, 
                            "City": target_city
                        },
                        kpi_data={
                            "Purchase Price": f"${purchase_price:,.0f}",
                            "Purchase Multiple": f"{ebitda_mult:.1f}x",
                            "Closing Leverage": f"{total_lev:.2f}x",
                            "Year 1 Cash ROI": f"{cash_roi:.1f}%",
                            "DSCR (Avg)": f"{deal_dscr:.2f}x",
                            "dscr": deal_dscr # Uses the base DSCR calculated in engine
                        },
                        sources_uses=s_and_u,
                        amort_schedule=pdf_amort_combined
                    )

                    # 4. Save to session state and trigger refresh
                    st.session_state.pdf_exported = True
                    st.session_state.temp_pdf = pdf_b 
                    st.rerun()

            # --- PHASE 2: PDF GENERATED -> SHOW GREYED OUT + DOWNLOAD ---
            else:
                st.button("✅ PDF Generated", use_container_width=True, disabled=True, key=f"exported_grey_{rc}")
                
                st.download_button(
                    label="⬇️ Click to Download Report",
                    data=st.session_state.temp_pdf,
                    file_name=f"Analysis_{target_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_final_{rc}"
                )

        with b3:
            if st.button("🔄 Clear & Restart Simulation", use_container_width=True, key=f"restart_btn_{rc}"):
                st.session_state.reset_counter += 1
                st.session_state.vault_saved = False
                st.session_state.pdf_exported = False
                # Clear temporary PDF storage
                if "temp_pdf" in st.session_state:
                    del st.session_state.temp_pdf
                # Clear run flag
                for k in list(st.session_state.keys()):
                    if "run_sim" in k: del st.session_state[k]
                st.rerun()

# ==========================================
# 8. MAIN APP NAVIGATION
# ==========================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Historical Performance", "Financial Projections", "Acquisition Simulation"], label_visibility="collapsed")
    st.sidebar.caption("v4.5.1 | Strategic M&A Ready")
    
    if page == "Home":
        show_home()
    elif page == "Historical Performance":
        show_history()
    elif page == "Financial Projections":
        show_projections()
    elif page == "Acquisition Simulation":
        show_acquisition()

if __name__ == "__main__":
    main()