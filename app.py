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

# ==========================================
# PDF ENGINE (FIXED KEYS)
# ==========================================
def create_pdf(rec, kpi_data, sources_uses=None, amort_schedule=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- PAGE 1: EXECUTIVE SUMMARY ---
    # (Header, Section 1: Identity, Section 2: KPIs, Section 3: Sources & Uses as previously built)
    pdf.set_fill_color(31, 73, 125) 
    pdf.set_text_color(255, 255, 255) 
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "STRATEGIC ACQUISITION SUMMARY", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(2)

    # 1. Identity & Verdict
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 1. Target Identity & Verdict", ln=True, fill=True)
    pdf.set_font("Arial", '', 10); pdf.ln(2)
    pdf.cell(95, 7, f"Target Name: {str(rec.get('Target', 'N/A'))}")
    pdf.cell(95, 7, f"Location: {str(rec.get('State', 'N/A'))}", ln=True)
    
    verdict = str(rec.get('Verdict', 'N/A')).upper()
    if "PASS" in verdict: pdf.set_text_color(39, 174, 96)
    elif "CAUTION" in verdict: pdf.set_text_color(241, 196, 15)
    else: pdf.set_text_color(231, 76, 60)
    pdf.set_font("Arial", 'B', 10); pdf.cell(0, 7, f"Final Verdict: {verdict}", ln=True)
    pdf.set_text_color(0, 0, 0); pdf.ln(4)

    # 2. Performance KPIs
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 2. Key Performance Indicators", ln=True, fill=True)
    pdf.set_font("Arial", '', 9); pdf.ln(2)
    for label, value in kpi_data.items():
        pdf.cell(110, 7, f" {str(label)}", border=1); pdf.cell(80, 7, f" {str(value)}", border=1, ln=True, align='C')
    pdf.ln(6)

    # 3. Sources & Uses
    if sources_uses:
        pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, " 3. Sources & Uses of Funds", ln=True, fill=True); pdf.ln(2)
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

    # 4. Strategic Summary
    pdf.set_font("Arial", 'B', 11); pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, " 4. Executive Strategic Summary", ln=True, fill=True)
    pdf.ln(2); pdf.set_font("Arial", '', 9)
    reasoning = str(rec.get('Reasons', 'No notes provided.')).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, reasoning)

    # --- PAGE 2: AMORTIZATION SCHEDULE ---
    if amort_schedule:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Appendix A: Debt Amortization Schedule", ln=True)
        pdf.set_font("Arial", '', 9); pdf.ln(5)
        
        # Table Header
        pdf.set_fill_color(31, 73, 125); pdf.set_text_color(255, 255, 255)
        pdf.cell(20, 8, "Year", 1, 0, 'C', 1); pdf.cell(40, 8, "Principal", 1, 0, 'C', 1)
        pdf.cell(40, 8, "Interest", 1, 0, 'C', 1); pdf.cell(40, 8, "Total Payment", 1, 0, 'C', 1)
        pdf.cell(50, 8, "Remaining Balance", 1, 1, 'C', 1)
        
        pdf.set_text_color(0, 0, 0)
        for row in amort_schedule:
            pdf.cell(20, 7, str(row['Year']), 1, 0, 'C')
            pdf.cell(40, 7, f"${row['Principal']:,.0f}", 1, 0, 'R')
            pdf.cell(40, 7, f"${row['Interest']:,.0f}", 1, 0, 'R')
            pdf.cell(40, 7, f"${row['Total Pmt']:,.0f}", 1, 0, 'R')
            pdf.cell(50, 7, f"${row['Balance']:,.0f}", 1, 1, 'R')

    response = pdf.output(dest='S')
    return bytes(response) if not isinstance(response, str) else response.encode('latin-1')

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
    # --- 0. FETCH HISTORICAL MV BASE DATA (2025) ---
    # Assuming your database has a 'year' or 'date' column
    try:
        # 1. Target the previous year relative to today (2025)
        sim_year_target = 2025 
        
        # 2. Query your historical database (adjust table/column names)
        # Using a sanitized filter to get the 2025 record
        mv_base_record = db_historical_data[db_historical_data['year'] == sim_year_target].iloc[0]
        
        # 3. Assign variables for KPI deltas
        mv_rev_base = mv_base_record['revenue']
        mv_ebitda_base = mv_base_record['ebitda']
        mv_fcf_base = mv_base_record['fcf']
        mv_margin_base = (mv_ebitda_base / mv_rev_base) * 100 if mv_rev_base > 0 else 0
        
    except (IndexError, KeyError, NameError):
        # Fallback if 2025 data isn't found to prevent app crash
        st.warning("⚠️ Historical 2025 data not found. Using zero-baseline for deltas.")
        mv_rev_base = 0
        mv_ebitda_base = 0
        mv_fcf_base = 0
        mv_margin_base = 0

    # --- 0. INITIALIZE GLOBAL DEFAULTS (Add this here!) ---
    apply_stress = False
    stress_retention_drop = 0.0
    stress_rate_jump = 0.0

    # --- 0. CRITICAL RESET & STATE LOGIC ---
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "📊 Financial Bridge"
        
    rc = st.session_state.reset_counter

    # --- 1. DATABASE PRE-LOAD ---
    db_file = "acquisition_history.csv"
    db_cols = ["Sim_ID", "Timestamp", "Simulator", "Target", "State", "City", "Verdict", "ROI", "Leverage", "Reasons"]
    if os.path.isfile(db_file):
        df_hist_master = pd.read_csv(db_file)
        df_hist_master["ROI_num"] = pd.to_numeric(df_hist_master["ROI"].astype(str).str.replace('%', ''), errors='coerce').fillna(0)
        df_hist_master["Lev_num"] = pd.to_numeric(df_hist_master["Leverage"].astype(str).str.lower().str.replace('x', ''), errors='coerce').fillna(0)
        df_hist_master["Timestamp_DT"] = pd.to_datetime(df_hist_master["Timestamp"], errors='coerce')
    else:
        df_hist_master = pd.DataFrame(columns=db_cols)

    st.title("🤝 Strategic Acquisition Simulation")

    # --- 2. INPUT BLOCKS ---
    st.markdown("<div class='section-header-blue'>👤 Simulator Identity</div>", unsafe_allow_html=True)
    id1, id2, id3 = st.columns(3)
    with id1: sim_first = st.text_input("First Name", value="", placeholder="Required", key=f"sf_{rc}")
    with id2: sim_last = st.text_input("Last Name", value="", placeholder="Required", key=f"sl_{rc}")
    with id3: sim_pos = st.selectbox("Position", ["", "CEO", "Director", "Analyst", "Consultant"], index=0, key=f"sp_{rc}")
    full_sim_name = f"{sim_first} {sim_last}".strip()

    # --- INVESTMENT MANDATE: PASS/FAIL CRITERIA ---
    st.markdown("<div class='section-header-blue'>⚖️ Investment Mandate: Pass/Fail Criteria</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1: target_max_lev_hurdle = st.number_input("Max Leverage Hurdle", 1.0, 10.0, value=None, placeholder="e.g. 3.5", key=f"h1_{rc}")
    with p2: target_min_ebitda_hurdle = st.number_input("Min Year 1 EBITDA ($)", 0, 100000000, value=None, placeholder="e.g. 1500000", key=f"h2_{rc}")
    with p3: target_max_price_hurdle = st.number_input("Max Multiple Hurdle (x)", 1.0, 25.0, value=None, placeholder="e.g. 7.0", key=f"h3_{rc}")
    with p4: target_available_cash = st.number_input("Available Cash ($)", 0, 500000000, value=None, placeholder="e.g. 5000000", key=f"h4_{rc}")

    p5, p6, p7 = st.columns(3)
    with p5: 
        target_min_dscr_hurdle = st.number_input("Min DSCR Hurdle", 0.0, 5.0, value=None, placeholder="e.g. 1.2", step=0.1, key=f"h5_{rc}")
    with p6: 
        roi_input_val = st.number_input("Min Year 1 ROI (%)", 0.0, 100.0, value=None, placeholder="e.g. 15.0", step=1.0, key=f"h6_{rc}")
        target_min_roi_hurdle = (roi_input_val / 100) if roi_input_val else 0
    with p7: 
        check_accretion = st.toggle("Hurdle: Deal Must Be Margin Accretive", value=False, key=f"h7_{rc}")

    # --- STEP 1: TARGET BASIC INFORMATION ---
    st.markdown("<div class='section-header-blue'>Step 1: Target Basic Information</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        target_name = st.text_input("Target Name", value="", placeholder="Enter target company name", key=f"tn_{rc}")
    with c2: 
        target_state = st.selectbox("State", ["", "Minnesota", "Wisconsin", "Iowa", "Illinois", "Florida"], index=0, key=f"ts_{rc}")
    with c3: 
        target_city = st.text_input("City", value="", placeholder="Enter city", key=f"tc_{rc}")
    with c4: 
        target_closing = st.date_input("Estimated Closing Date", value=None, min_value=date.today(), key=f"td_{rc}")

    st.markdown("<div class='section-header-blue'>Step 2: Financial Assumptions</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: 
        t_rev = st.number_input("Target TTM Revenue ($)", value=None, placeholder="Enter Revenue", key=f"f1a_{rc}")
        t_ebitda = st.number_input("Target TTM EBITDA ($)", value=None, placeholder="Enter EBITDA", key=f"f1b_{rc}")
    with f2:
        # Change 90 to 0 and 5 to 0
        retention_rate = st.slider("Retention (%)", 0, 100, 0, key=f"f2a_{rc}") / 100
        organic_growth = st.slider("Year 1 Growth (%)", -50, 100, 0, key=f"f2b_{rc}") / 100
    with f3:
        # Change 5.0 to 0.0 and 10 to 0
        ebitda_mult = st.slider("Purchase Multiple (x)", 0.0, 20.0, 0.0, 0.1, key=f"f3a_{rc}")
        synergy_pct = st.slider("Synergy Potential (%)", -50, 100, 0, key=f"f3b_{rc}") / 100
    with f4:
        # Change 75 to 0
        fcf_conv = st.slider("FCF Conversion (%)", 0, 100, 0, key=f"f4a_{rc}") / 100
        purchase_price = (t_ebitda * ebitda_mult) if t_ebitda else 0
        st.metric("Implied Purchase Price", f"${purchase_price:,.0f}")

    # --- STEP 3: FINANCING (CLEAN START VERSION) ---
    st.markdown("<div class='section-header-blue'>Step 3: Financing</div>", unsafe_allow_html=True)
    fi1, fi2, fi3, fi4 = st.columns(4)
    with fi1:
        # Reset 50 to 0 and 10 to 0
        debt_pct = st.slider("Bank Loan (%)", 0, 100, 0, key=f"fi1a_{rc}") / 100
        seller_pct = st.slider("Seller Note (%)", 0, 100, 0, key=f"fi1b_{rc}") / 100
    with fi2:
        # Use value=None instead of 8.5
        int_rate_input = st.number_input("Interest Rate (%)", value=None, placeholder="e.g. 8.5", key=f"fi2a_{rc}")
        # Reset Term slider to 1
        loan_term = st.slider("Term (Years)", 1, 25, 1, key=f"fi2b_{rc}")
        transaction_fees = st.number_input("Est. Transaction Fees ($)", value=None, placeholder="e.g. 50000", key=f"fi2c_{rc}")
    
    # --- ENGINE PRE-CALCULATIONS (Safe Math) ---
    # Convert interest rate input to decimal and sanitize
    int_rate_val = (int_rate_input / 100) if int_rate_input is not None else 0.0
    
    # These 's_' variables ensure the app doesn't crash from NoneType errors
    s_t_ebitda = t_ebitda if t_ebitda is not None else 0
    s_purchase_price = s_t_ebitda * ebitda_mult
    s_fees = transaction_fees if transaction_fees is not None else 0
    
    # Core Financing Totals
    new_debt = s_purchase_price * debt_pct
    s_seller_note = s_purchase_price * seller_pct
    total_financing = new_debt + s_seller_note
    cash_equity_needed = (s_purchase_price + s_fees) - total_financing

    with fi3: 
        st.metric("Total Financing", f"${total_financing:,.0f}")
    with fi4: 
        st.metric("Cash Equity", f"${cash_equity_needed:,.0f}")
    
    # --- ENGINE PRE-CALCULATIONS (Sanitization & Global Variables) ---
    s_t_ebitda = t_ebitda if t_ebitda is not None else 0
    s_purchase_price = s_t_ebitda * ebitda_mult
    s_fees = transaction_fees if transaction_fees is not None else 0
    s_new_debt = s_purchase_price * debt_pct
    s_seller_note = s_purchase_price * seller_pct
    total_financing = s_new_debt + s_seller_note
    cash_equity_needed = (s_purchase_price + s_fees) - total_financing

    with fi3: st.metric("Total Financing", f"${total_financing:,.0f}")
    with fi4: st.metric("Cash Equity", f"${cash_equity_needed:,.0f}")

    # --- UNIFIED READINESS CALCULATION ---
    # Logic updated to require non-zero/non-None values for 0% initial state
    checklist = {
        "Identity": sim_first != "" and sim_last != "",
        "Mandates": all(v is not None for v in [target_max_lev_hurdle, target_min_ebitda_hurdle, target_max_price_hurdle]),
        "Target Info": target_name != "" and target_state != "",
        "Financials": t_ebitda is not None and ebitda_mult > 0, # ebitda_mult must be moved from 0
        "Financing": int_rate_input is not None and debt_pct > 0, # Interest must be typed
    }
    
    progress_pct = sum(checklist.values()) / len(checklist)
    bar_color = "red" if progress_pct < 0.5 else "orange" if progress_pct < 1.0 else "green"
    
    st.markdown(f"### 📋 Simulation Readiness: <span style='color:{bar_color};'>{int(progress_pct*100)}%</span>", unsafe_allow_html=True)
    st.progress(progress_pct)

    # --- SIMULATION STATE MANAGEMENT ---
    if f"run_sim_{rc}" not in st.session_state:
        st.session_state[f"run_sim_{rc}"] = False
    
    # Define sim_is_active clearly for all downstream logic
    sim_is_active = st.session_state[f"run_sim_{rc}"]
    
    # Define is_locked: True if any required field is incomplete
    is_locked = progress_pct < 1.0

    st.divider()
    
    # --- ANALYSIS GATE ---
    if is_locked:
        st.info("💡 **Simulator Guard:** Complete all fields to unlock analysis.")
        st.button("🚀 Run Analysis", disabled=True, key=f"btn_dis_{rc}")
    else:
        if st.button("🚀 Run Analysis", type="primary", key=f"run_btn_{rc}"):
            st.session_state[f"run_sim_{rc}"] = True
            st.rerun() # Ensure UI updates immediately to show new sections

    # ==============================================================================
    # THE OUTPUT GATE: Everything below only shows if Analysis is active
    # ==============================================================================
    if sim_is_active:
        # --- 4. SCENARIO STRESS TESTING (Now properly nested and gated) ---
        st.markdown("### 🧪 Scenario Stress Testing")
        
        # This section is now only visible if sim_is_active is True
        with st.expander("Configure Shock Scenario Parameters", expanded=False):
            c_stress1, c_stress2 = st.columns(2)
            with c_stress1:
                stress_retention_drop = st.number_input(
                    "Retention Shock (% Drop)", 
                    value=10.0, step=1.0, 
                    key=f"str_ret_{rc}"
                ) / 100
            with c_stress2:
                stress_rate_jump = st.number_input(
                    "Interest Rate Shock (bps Rise)", 
                    value=200.0, step=50.0, 
                    key=f"str_rate_{rc}"
                ) / 10000
        
            apply_stress = st.toggle("🚨 Apply Stress Test to Results", value=False, key=f"str_tg_{rc}")
            
            if apply_stress:
                st.warning("⚠️ **Scenario Mode:** Pro-forma results now reflect the shock parameters.")

    # --- 5. GLOBAL ENGINE MATH ---
    # SANITIZE INPUTS
    s_t_rev = t_rev if t_rev is not None else 0
    s_new_debt = s_new_debt if s_new_debt is not None else 0
    s_seller_note = s_seller_note if s_seller_note is not None else 0
    current_total_debt = s_new_debt + s_seller_note

    # BASE DATA 
    # --- 0. FETCH 2025 HISTORICAL DATA ---
    try:
    # Target 2025 data from your historical dataframe
        mv_2025 = db_historical_data[db_historical_data['year'] == 2025].iloc[0]
        mv_rev_base = mv_2025['revenue']
        mv_ebitda_base = mv_2025['ebitda']
        mv_fcf_base = mv_2025['fcf']
        mv_margin_base = (mv_ebitda_base / mv_rev_base) * 100 if mv_rev_base > 0 else 0
    except:
    # Fallback to avoid NameErrors
        mv_rev_base, mv_ebitda_base, mv_fcf_base, mv_margin_base = 0, 0, 0, 0

    # 1. BASE CASE MATH (Permanent Record)
    base_adj_t_ebitda = s_t_ebitda * retention_rate
    base_consolidated_ebitda = (mv_ebitda_base + base_adj_t_ebitda) * (1 + organic_growth) + (base_adj_t_ebitda * synergy_pct)
    base_annual_debt_svc = (s_new_debt / (loan_term if loan_term > 0 else 1)) + (s_new_debt * int_rate_val)
    base_net_fcf = (base_consolidated_ebitda * fcf_conv) - base_annual_debt_svc
    base_roi = (base_net_fcf / (cash_equity_needed if cash_equity_needed > 0 else 1) * 100)
    base_leverage = (current_total_debt + 1500000) / (base_consolidated_ebitda if base_consolidated_ebitda != 0 else 1)

    # 2. ACTIVE DISPLAY MATH (Used for KPIs and Proximity)
    active_shock = apply_stress and sim_is_active
    calc_retention = retention_rate - stress_retention_drop if active_shock else retention_rate
    calc_int_rate = int_rate_val + stress_rate_jump if active_shock else int_rate_val
    
    adj_t_ebitda = s_t_ebitda * calc_retention
    synergy_val = adj_t_ebitda * synergy_pct
    growth_val = (mv_ebitda_base + adj_t_ebitda) * organic_growth
    consolidated_ebitda = (mv_ebitda_base + adj_t_ebitda) + growth_val + synergy_val
    
    # Define Metrics needed for Proximity (Crucial Order)
    total_leverage = (current_total_debt + 1500000) / (consolidated_ebitda if consolidated_ebitda != 0 else 1)
    annual_debt_svc = (s_new_debt / (loan_term if loan_term > 0 else 1)) + (s_new_debt * calc_int_rate)
    deal_dscr = consolidated_ebitda / annual_debt_svc if annual_debt_svc > 0 else 0
    net_fcf = (consolidated_ebitda * fcf_conv) - annual_debt_svc
    cash_roi = (net_fcf / (cash_equity_needed if cash_equity_needed > 0 else 1) * 100)

    # --- 5c. PROXIMITY & WEAKEST LINK (Sanitized) ---
    # We use 'or 0' to handle None values gracefully
    s_h_lev = target_max_lev_hurdle or 0
    s_h_dscr = target_min_dscr_hurdle or 0
    s_h_roi = target_min_roi_hurdle or 0

    prox_leverage = (s_h_lev - total_leverage) / s_h_lev if s_h_lev != 0 else 0
    prox_dscr = (deal_dscr - s_h_dscr) / s_h_dscr if s_h_dscr != 0 else 0
    prox_roi = (cash_roi - s_h_roi) / s_h_roi if s_h_roi != 0 else 0

    proximities = {"Leverage": prox_leverage, "DSCR": prox_dscr, "ROI": prox_roi}
    weakest_link = min(proximities, key=lambda k: abs(proximities[k]))
    prox_val = proximities[weakest_link] * 100

    # 4. CAPACITY & REVENUE
    # Use the sanitized hurdle defined in the Verdict Logic section
    s_h_lev = target_max_lev_hurdle or 0

    max_debt_allowed = (consolidated_ebitda * s_h_lev) - 1500000
    debt_capacity_margin = max_debt_allowed - current_total_debt
    
    # Pro-forma Revenue & Margin
    pf_rev_y1 = (mv_rev_base + s_t_rev) * (1 + organic_growth)
    pf_margin_y1 = (consolidated_ebitda / pf_rev_y1 * 100) if pf_rev_y1 > 0 else 0

   # --- VERDICT LOGIC (Sanitized for NoneTypes) ---
    fail_reasons = []
    caution_reasons = []

    # 1. Sanitize hurdles: Use 'or 0' to handle None values gracefully
    s_h_lev = target_max_lev_hurdle or 0
    s_h_mult = target_max_price_hurdle or 0
    s_h_dscr = target_min_dscr_hurdle or 0
    s_h_cash = target_available_cash or 0
    s_h_roi = roi_input_val or 0

    # 2. FAIL LOGIC: Only trigger if the hurdle is actually set (> 0)
    # This prevents 'NoneType' comparison errors and premature fails
    if s_h_lev > 0 and total_leverage > s_h_lev: 
        fail_reasons.append("Leverage Gap")
    if s_h_mult > 0 and ebitda_mult > s_h_mult: 
        fail_reasons.append("Over Valuation")
    if s_h_dscr > 0 and deal_dscr < s_h_dscr: 
        fail_reasons.append("Insolvent Coverage")
    if s_h_cash > 0 and cash_equity_needed > s_h_cash: 
        fail_reasons.append("Insufficient Cash")

    # 3. CAUTION LOGIC (None-Safe)
    if not fail_reasons:
        if s_h_lev > 0 and total_leverage > (s_h_lev * 0.9): 
            caution_reasons.append("High Leverage Warning")
        if s_h_dscr > 0 and deal_dscr < (s_h_dscr * 1.1): 
            caution_reasons.append("Tight Debt Service")
        if s_h_roi > 0 and cash_roi < (s_h_roi * 1.1): 
            caution_reasons.append("Low ROI Margin")

    # 4. FINAL VERDICT ASSIGNMENT
    if fail_reasons:
        verdict, v_color, action = "FAIL: DE-PRIORITIZE DEAL", "#e74c3c", "Action: Terminate Deal"
        reason_str = "Critical Barriers: " + " | ".join(fail_reasons)
    elif caution_reasons:
        verdict, v_color, action = "CAUTION: MARGINAL FIT", "#f1c40f", "Action: Re-negotiate Price"
        reason_str = "Warning Zones: " + " | ".join(caution_reasons)
    else:
        verdict, v_color, action = "PASS: STRATEGIC FIT", "#27ae60", "Action: Proceed with LOI"
        reason_str = "All investment mandates cleared."

    # --- OUTPUT DISPLAY ---
    if sim_is_active:
        # 1. Pre-calculate all impacts to prevent NameErrors
        margin_impact = pf_margin_y1 - mv_margin_base
        fcf_impact = net_fcf - mv_fcf_base

        # 2. Verdict Banner
        with st.spinner("Processing strategic pro-forma..."):
            st.markdown(f"""
                <div style='background-color:{v_color}; padding:25px; border-radius:10px; text-align:center; color:white; margin-bottom:20px;'>
                    <h2 style='color:white; margin:0;'>{"⚠️ STRESSED: " if apply_stress else ""}{verdict}</h2>
                    <p style='color:white; font-weight:bold; margin:5px 0;'>{action}</p>
                    <p style='color:white; font-size:0.9em; margin:0;'>{reason_str}</p>
                </div>
            """, unsafe_allow_html=True)

            # 3. KPI Grid Row 1: Primary Debt & Return Metrics
            k1, k2, k3, k4 = st.columns(4)
            with k1: 
                st.metric("DSCR", f"{deal_dscr:.2f}x", 
                          delta=f"{deal_dscr - (s_h_dscr or 0):.2f} vs Hurdle")
            with k2: 
                st.metric("Combined EBITDA", f"${consolidated_ebitda:,.0f}", 
                          delta=f"${consolidated_ebitda - (target_min_ebitda_hurdle or 0):,.0f} vs Goal")
            with k3: 
                st.metric("Net Free Cash Flow", f"${net_fcf:,.0f}", 
                          delta=f"${fcf_impact:,.0f} vs 2025 Base")
            with k4: 
                st.metric("Cash ROI", f"{cash_roi:.1f}%", 
                          delta=f"{cash_roi - (s_h_roi * 100):.1f}% vs Goal")

            # 4. KPI Grid Row 2: Leverage & Operations
            k5, k6, k7, k8 = st.columns(4)
            with k5: 
                st.metric("Debt / EBITDA", f"{total_leverage:.2f}x", 
                          delta=f"{total_leverage - (s_h_lev or 0):.2f} vs Hurdle", delta_color="inverse")
            with k6: 
                # margin_impact is now safely defined above
                st.metric("Margin Profile", f"{pf_margin_y1:.1f}%", 
                          delta=f"{margin_impact:+.1f}% vs 2025 Base")
            with k7: 
                st.metric("Cash Required", f"${cash_equity_needed:,.0f}", 
                          delta=f"${(s_h_cash or 0) - cash_equity_needed:,.0f} Budget")
            with k8: 
                st.metric("EBITDA Multiple", f"{ebitda_mult:.1f}x", 
                          delta=f"{ebitda_mult - (s_h_mult or 0):.1f} vs Limit", delta_color="inverse")

     # --- 6. TABS NAVIGATION (FIXED ORDER) ---
    st.divider()
    
    # 1. Define the list FIRST to avoid NameError
    tabs_list = ["📊 Financial Bridge", "📈 Amortization", "🧪 Sensitivity Analysis", "📝 Record Management"]

    # 2. Initialize the session state if it doesn't exist
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = tabs_list[0]

    # 3. Now call the radio widget using the defined tabs_list
    st.session_state.active_tab = st.radio(
        "Simulation Navigation", 
        options=tabs_list, 
        horizontal=True, 
        label_visibility="collapsed", 
        key=f"nav_bar_final_{rc}" 
    )
    st.divider()  
 
    # 3. TAB CONTENT LOGIC
    if st.session_state.active_tab == "📊 Financial Bridge":
        st.markdown("### EBITDA Bridge: 2025 Base to Pro-Forma")
        # Logic to visualize the value-add of the acquisition
        bridge_data = {
            "2025 MV Base": mv_ebitda_base,
            "Acquisition EBITDA": s_t_ebitda,
            "Estimated Synergies": (s_t_ebitda * synergy_pct),
            "Pro-Forma Total": consolidated_ebitda
        }
        # Use bar_chart for immediate visualization
        st.bar_chart(bridge_data)

    elif st.session_state.active_tab == "📈 Amortization":
        amort_data = []
        rem_bal = new_debt
        ann_pri = new_debt / loan_term
        for y in range(1, loan_term + 1):
            i_pmt = rem_bal * int_rate_val
            t_pmt = ann_pri + i_pmt
            rem_bal -= ann_pri
            amort_data.append({"Year": f"Y{y}", "Principal": ann_pri, "Interest": i_pmt, "Total Pmt": t_pmt, "Balance": max(0, rem_bal)})
            df_amort = pd.DataFrame(amort_data)
            st.dataframe(df_amort.set_index("Year").style.format("${:,.0f}"), use_container_width=True)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_amort.to_excel(writer, index=False, sheet_name='Amortization')
        st.download_button("📥 Export Schedule", buffer.getvalue(), f"Amortization_{target_name}.xlsx", key=f"dl_am_{rc}")
        
        st.divider()
        st.markdown("#### Debt Service Coverage Trend")
        df_amort["DSCR_Safety"] = consolidated_ebitda / df_amort["Total Pmt"]
        fig_dscr = px.line(df_amort, x="Year", y="DSCR_Safety", title="Projected DSCR Over Loan Term", markers=True)
        fig_dscr.add_hline(y=target_min_dscr_hurdle, line_dash="dash", line_color="red", annotation_text="Lender Minimum Hurdle")
        fig_dscr.update_layout(yaxis_title="DSCR Ratio (x)", template="plotly_white")
        st.plotly_chart(fig_dscr, use_container_width=True)

    elif st.session_state.active_tab == "🧪 Sensitivity Analysis":
        st.markdown("### ROI Sensitivity Analysis")
        
        # --- FIXED ALIGNMENT BLOCK ---
        # Ensure these lines match the indentation of the 'st.markdown' above
        c1, c2 = st.columns([3, 1])
        
        with c2:
            st.info("💡 ROI Sensitivity: This heatmap visualizes how Year 1 Cash ROI changes relative to Purchase Multiple and Interest Rate shifts.")

        with c1:
            # Logic for generating a sensitivity matrix
            multiples = [ebitda_mult * (1 + x) for x in [-0.2, -0.1, 0, 0.1, 0.2]]
            rates = [int_rate_val * (1 + x) for x in [-0.2, -0.1, 0, 0.1, 0.2]]
            
            # Sensitivity calculation loop
            sens_data = []
            for m in multiples:
                row = []
                for r in rates:
                    # Calculate theoretical ROI based on variable shifts
                    temp_price = s_t_ebitda * m
                    temp_debt = temp_price * debt_pct
                    temp_interest = temp_debt * r
                    temp_fcf = (s_t_ebitda * fcf_conv) - temp_interest
                    temp_equity = (temp_price + s_fees) - (temp_debt + (temp_price * seller_pct))
                    temp_roi = (temp_fcf / temp_equity * 100) if temp_equity > 0 else 0
                    row.append(temp_roi)
                sens_data.append(row)

            # Display as a formatted dataframe heatmap
            df_sens = pd.DataFrame(
                sens_data, 
                index=[f"{m:.1f}x" for m in multiples], 
                columns=[f"{r*100:.1f}%" for r in rates]
            )
            st.write("Vertical: Purchase Multiple | Horizontal: Interest Rate")
            st.dataframe(df_sens.style.background_gradient(cmap='RdYlGn').format("{:.1f}%"))

        st.divider()
        st.markdown("#### 2. Leverage Sensitivity (Lender Risk Analysis)")
        l1, l2 = st.columns([3, 1])
        with l1:
            def calc_lev_sens(m, d_p):
                debt_amt = (t_ebitda * m) * d_p
                return round((debt_amt + 1500000) / (consolidated_ebitda if consolidated_ebitda != 0 else 1), 2)
            
            m_rng_lev = [ebitda_mult + i for i in [1, 0.5, 0, -0.5, -1]]
            d_pct_rng = [debt_pct + i for i in [-0.2, -0.1, 0, 0.1, 0.2]]
            sens_lev = pd.DataFrame([[calc_lev_sens(m, d) for d in d_pct_rng] for m in m_rng_lev], 
                                index=[f"{m:.1f}x" for m in m_rng_lev], 
                                columns=[f"{int(d*100)}%" for d in d_pct_rng])
            
            fig_lev = px.imshow(sens_lev, text_auto=True, color_continuous_scale="Reds", 
                            color_continuous_midpoint=target_max_lev_hurdle, title="Total Leverage vs. Debt % and Multiples")
            fig_lev.update_layout(xaxis_title="Bank Debt %", yaxis_title="Multiple (x)")
            st.plotly_chart(fig_lev, use_container_width=True)
        with l2:
            st.warning(f"**Lender Risk:** Max Leverage Hurdle: {target_max_lev_hurdle}x.")

        st.divider()
        st.subheader("💡 Strategic Recommendation")
        if cash_roi >= roi_input_val and total_leverage <= target_max_lev_hurdle:
            st.success("The current structure is **Optimized**. Maximizes leverage while exceeding ROI goals.")
        elif total_leverage > target_max_lev_hurdle:
            st.error(f"**Action Required:** Deal is over-leveraged. Reduce Bank Debt or Multiple.")
        else:
            st.info("**Optimization Opportunity:** Additional 'Debt Capacity' exists. Could increase Bank Debt to preserve cash.")

        st.divider()
        st.subheader("🏦 Debt Capacity & Margin of Safety")
        max_debt_allowed = (consolidated_ebitda * target_max_lev_hurdle) - 1500000
        current_new_debt = s_new_debt + s_seller_note
        remaining_capacity = max_debt_allowed - current_new_debt
        
        cap_c1, cap_c2 = st.columns([2, 1])
        with cap_c1:
            fig_cap = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", value = current_new_debt,
                    title = {'text': "Used vs. Max Debt Capacity ($)", 'font': {'size': 18}},
                    delta = {'reference': max_debt_allowed, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, max(max_debt_allowed * 1.2, 1)], 'tickformat': "$,.0f"},
                        'bar': {'color': "#1f497d"},
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_debt_allowed}
                    }
                ))
                # Update layout and display the figure
            fig_cap.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_cap, use_container_width=True)

        with cap_c2:
            st.markdown("#### Capital Observations")
            if remaining_capacity > 0:
                st.write(f"✅ **Dry Powder:** ${remaining_capacity:,.0f} available.")
            else:
                st.write(f"🚨 **Over-Leveraged:** ${abs(remaining_capacity):,.0f} over limit.")

    elif st.session_state.active_tab == "📝 Record Management":
        if not df_hist_master.empty:
            # --- 1. SEARCH & FILTER SECTION ---
            with st.form("vault_search_form"):
                f1, f2, f3 = st.columns(3)
                with f1: 
                    v_filt = st.multiselect("Verdict", options=df_hist_master["Verdict"].unique(), default=list(df_hist_master["Verdict"].unique()))
                with f2: 
                    t_filt = st.text_input("Target Search")
                    city_filt = st.text_input("City Search")
                with f3: 
                    d_range = st.date_input("Date Range", value=(df_hist_master["Timestamp_DT"].min().date(), date.today()))
                
                if st.form_submit_button("🔍 Run Search"):
                    mask = (df_hist_master["Verdict"].isin(v_filt))
                    if t_filt: mask = mask & (df_hist_master["Target"].str.contains(t_filt, case=False))
                    if city_filt: mask = mask & (df_hist_master["City"].str.contains(city_filt, case=False))
                    st.session_state.filtered_vault = df_hist_master[mask]

            display_df = st.session_state.get("filtered_vault", df_hist_master)
            st.dataframe(display_df.drop(columns=["ROI_num", "Lev_num", "Timestamp_DT"], errors='ignore').set_index("Sim_ID"), use_container_width=True)

            st.divider()
            
    # --- 2. RECOVERY & DELETION ACTIONS ---
        c1, c2 = st.columns(2)
            
        with c1:
            st.markdown("### 📄 Recover Historical Memo")
            sel_rec_recov = st.selectbox("Select Record to Recover:", 
                                        options=display_df.apply(lambda x: f"ID: {x['Sim_ID']} | {x['Target']}", axis=1), 
                                        key=f"recovery_sel_{rc}")
                
            if st.button("📑 Generate Updated PDF Structure"):
                sid = int(sel_rec_recov.split("|")[0].replace("ID: ", "").strip())
                row = df_hist_master[df_hist_master["Sim_ID"] == sid].iloc[0]
                    
                try:
                    # Map historical CSV columns to the IDENTITY record (Target info)
                    hist_ident = {
                        "Target": str(row["Target"]), 
                        "Simulator": str(row["Simulator"]), 
                        "State": str(row["State"]), 
                        "Verdict": str(row["Verdict"]),
                        "Reasons": str(row["Reasons"]) # Rationale/Audit Trail
                    }
                        
                    # Re-construct the KPI Table dictionary for the PDF engine
                    hist_kpis = {
                        "Cash ROI": str(row["ROI"]),
                        "Leverage (Debt/EBITDA)": str(row["Leverage"]),
                        "City": str(row.get("City", "N/A")),
                        "Timestamp": str(row["Timestamp"])
                    }
                        
                    # Call the latest PDF engine with synced data
                    pdf_bytes = create_pdf(hist_ident, hist_kpis)
                        
                    st.download_button(
                        label="📥 Download Recovered PDF",
                        data=pdf_bytes,
                        file_name=f"Historical_Memo_{row['Target'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key=f"dl_hist_{sid}"
                    )
                except Exception as e: 
                    st.error(f"Error mapping historical data to PDF: {e}")

        with c2:
            st.markdown("### 🗑️ Data Housekeeping")
            sel_rec_del = st.selectbox("Select Record to Permanent Delete:", 
                                        options=display_df.apply(lambda x: f"ID: {x['Sim_ID']} | {x['Target']}", axis=1), 
                                        key=f"delete_sel_{rc}")
                
            if st.button("❌ Delete Selected Record", type="secondary"):
                sid_del = int(sel_rec_del.split("|")[0].replace("ID: ", "").strip())
                # Update the master dataframe and save to CSV
                df_updated = df_hist_master[df_hist_master["Sim_ID"] != sid_del]
                df_updated.to_csv(db_file, index=False)
                st.warning(f"Record ID {sid_del} has been deleted.")
                st.rerun()
    else:
        st.info("No records found in the acquisition vault.")

    # --- FINAL EXECUTIVE WRAPPER (Indented under if sim_is_active) ---
    st.markdown("---")
    with st.expander("📝 View Executive Deal Memo Summary", expanded=False):
        memo_c1, memo_c2 = st.columns(2)
        with memo_c1:
            st.write(f"**Target:** {target_name} ({target_state})")
            st.write(f"**Enterprise Value:** ${purchase_price:,.0f} ({ebitda_mult}x)")
            st.write(f"**Transaction Costs:** ${transaction_fees:,.0f}")
        with memo_c2:
            # These values display current UI state (reflects stress if toggled)
            st.write(f"**Pro-Forma EBITDA:** ${consolidated_ebitda:,.0f}")
            st.write(f"**Year 1 Cash ROI:** {cash_roi:.1f}%")
            st.write(f"**Closing Leverage:** {total_leverage:.2f}x")
        st.info(f"**Final Verdict:** {verdict}")

    st.divider()
    st.markdown("### 📑 Finalize Deal Memo & Database Entry")
        
    # Audit Warning: Ensures user knows what is being saved
    if apply_stress:
        st.warning("⚠️ **Database Sync Alert:** Stress Test is ACTIVE. The system will save the **Base Case** (Pre-Shock) results to the Acquisition Vault to maintain historical data integrity.")
    else:
        st.info("ℹ️ The database will record the current **Base Case** results and your strategic rationale.")

    st.caption("Tip: Press **Ctrl + Enter** to apply your notes before downloading.")
        
    strat_notes = st.text_area(
        "Strategic Deal Memo", 
        placeholder="Enter strategic rationale, synergy details, and risk mitigation...", 
        key=f"memo_{rc}"
    )

    # UI State Confirmation for the Memo
    if strat_notes:
        st.markdown("✅ *Memo content accepted and ready for export.*")
    else:
        st.markdown("⚪ *Memo is currently empty.*")
        
    b1, b2, b3 = st.columns(3)
        
    with b1:
        if st.button("📁 Save Base Case to Vault", use_container_width=True, key=f"save_{rc}"):
            # Audit trail: 'weakest_link' and 'prox_val' must be defined in the Global Engine Math
            full_audit = f"BASE CASE ENTRY: {strat_notes} | Critical Risk: {weakest_link} @ {prox_val:+.1f}%"
                
            new_entry = {
                "Sim_ID": len(df_hist_master)+1, 
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), 
                "Simulator": full_sim_name,
                "Target": target_name, 
                "City": target_city,
                "State": target_state,
                "Verdict": verdict.split(":")[0],
                "ROI": f"{base_roi:.1f}%", 
                "Leverage": f"{base_leverage:.2f}x",
                "Reasons": full_audit
            }
            pd.concat([df_hist_master, pd.DataFrame([new_entry])]).to_csv(db_file, index=False)
            st.success("✅ Base Case saved to database.")

    with b2:
        # Generate PDF Summary (Reflects current UI state)
        if st.button("📄 Generate PDF Summary", use_container_width=True, key=f"pdf_btn_{rc}"):
            if not strat_notes:
                st.warning("⚠️ Please enter strategic notes before generating the report.")
            else:
                try:
                    # 1. Identity & Narrative
                    live_pdf_data = {
                        "Target": str(target_name), 
                        "Simulator": str(full_sim_name), 
                        "State": str(target_state), 
                        "Verdict": f"{verdict} ({action})", 
                        "Reasons": f"STRATEGIC RATIONALE:\n{strat_notes}\n\nSENSITIVITY ALERT:\n{weakest_link} is {prox_val:+.1f}% from hurdle."
                    }
                        
                    # 2. Key Performance Indicators
                    performance_kpis = {
                        "Enterprise Value": f"${purchase_price:,.0f}",
                        "Purchase Multiple": f"{ebitda_mult}x",
                        "Transaction Fees": f"${transaction_fees:,.0f}",
                        "Total Cash Equity Needed": f"${cash_equity_needed:,.0f}",
                        "Pro-Forma EBITDA (Y1)": f"${consolidated_ebitda:,.0f}",
                        "Debt / EBITDA (Leverage)": f"{total_leverage:.2f}x",
                        "Cash ROI (Year 1)": f"{cash_roi:.1f}%",
                        "Debt Capacity Margin": f"${debt_capacity_margin:,.0f}"
                    }

                    # 3. Sources & Uses Table
                    sources_and_uses = {
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

                    # 4. Amortization Schedule (Internal calculation for scope safety)
                    pdf_amort_data = []
                    temp_bal = s_new_debt
                    temp_pri = s_new_debt / (loan_term if loan_term > 0 else 1)
                    for y in range(1, loan_term + 1):
                        i_pmt = temp_bal * calc_int_rate
                        t_pmt = temp_pri + i_pmt
                        temp_bal -= temp_pri
                        pdf_amort_data.append({
                            "Year": f"Y{y}", "Principal": temp_pri, 
                            "Interest": i_pmt, "Total Pmt": t_pmt, "Balance": max(0, temp_bal)
                        })

                    # 5. Build PDF
                    pdf_bytes = create_pdf(
                        live_pdf_data, 
                        performance_kpis, 
                        sources_uses=sources_and_uses, 
                        amort_schedule=pdf_amort_data
                    )
                        
                    st.download_button(
                        label="📥 Download Full Strategic Report", 
                        data=pdf_bytes, 
                        file_name=f"MagnaVita_Report_{target_name}.pdf", 
                        mime="application/pdf", 
                        key=f"dl_pdf_{rc}"
                    )
                except Exception as e: 
                    st.error(f"PDF Generation Error: {e}")

    with b3:
        if st.button("🔄 Restart & Clear All", use_container_width=True, key=f"restart_btn_{rc}"):
            # Clear all specific simulation keys from memory
            keys_to_clear = [k for k in st.session_state.keys() if any(x in k for x in ["sf_", "sl_", "sp_", "h", "tn_", "ts_", "tc_", "td_", "f", "fi", "memo", "str_", "run_sim"])]
            for key in keys_to_clear:
                del st.session_state[key]
            
            # Increment reset counter to rotate widget keys
            st.session_state.reset_counter += 1 
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