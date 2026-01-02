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
def create_pdf(rec, kpi_data):
    pdf = FPDF()
    pdf.add_page()
    
    # --- BLUE CORPORATE HEADER ---
    pdf.set_fill_color(31, 73, 125) 
    pdf.set_text_color(255, 255, 255) 
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "STRATEGIC ACQUISITION SUMMARY", ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='R')
    pdf.ln(5)

    # --- SECTION 1: DEAL IDENTITY ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240) 
    pdf.cell(0, 10, " 1. Target Identity", ln=True, fill=True)
    pdf.set_font("Arial", '', 11)
    pdf.ln(2)
    pdf.cell(95, 8, f"Target Name: {str(rec.get('Target', 'N/A'))}")
    pdf.cell(95, 8, f"Simulator: {str(rec.get('Simulator', 'N/A'))}", ln=True)
    pdf.cell(95, 8, f"Location: {str(rec.get('State', 'N/A'))}")
    pdf.cell(95, 8, f"Final Verdict: {str(rec.get('Verdict', 'N/A'))}", ln=True)
    pdf.ln(8)

    # --- SECTION 2: PERFORMANCE TABLE ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, " 2. Key Performance Indicators", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(110, 10, " Financial Metric", border=1, fill=True)
    pdf.cell(80, 10, " Result", border=1, fill=True, ln=True, align='C')
    
    pdf.set_font("Arial", '', 10)
    for label, value in kpi_data.items():
        if label != "Reasons":
            pdf.cell(110, 10, f" {str(label)}", border=1)
            pdf.cell(80, 10, f" {str(value)}", border=1, ln=True, align='C')
    pdf.ln(8)

    # --- SECTION 3: STRATEGIC REASONING ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, " 3. Decision Rationale", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 10)
    reasoning = str(rec.get('Reasons', 'No notes provided.'))
    pdf.multi_cell(0, 8, f"Observations: {reasoning}")
    
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Magna Vita Strategic Planning - Internal Use Only", align='C')
    
    # FIX: Return bytes directly via a string output buffer
    return pdf.output(dest='S').encode('latin-1')

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
        
# ==========================================
# # 7. VIEW: ACQUISITION SIMULATION (FINAL INTEGRATED VERSION)
# ==============================================================================
def show_acquisition():
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

    st.markdown("<div class='section-header-blue'>⚖️ Investment Mandate: Pass/Fail Criteria</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1: target_max_lev_hurdle = st.number_input("Max Leverage Hurdle", 1.0, 10.0, value=3.5, step=0.1, key=f"h1_{rc}")
    with p2: target_min_ebitda_hurdle = st.number_input("Min Year 1 EBITDA ($)", 0, 100000000, value=1500000, step=50000, key=f"h2_{rc}")
    with p3: target_max_price_hurdle = st.number_input("Max Multiple Hurdle (x)", 1.0, 25.0, value=7.0, step=0.1, key=f"h3_{rc}")
    with p4: target_available_cash = st.number_input("Available Cash ($)", 0, 500000000, value=5000000, step=100000, key=f"h4_{rc}")

    p5, p6, p7 = st.columns(3)
    with p5: target_min_dscr_hurdle = st.number_input("Min DSCR Hurdle", value=1.2, step=0.1, key=f"h5_{rc}")
    with p6: 
        roi_input_val = st.number_input("Min Year 1 ROI (%)", value=15.0, step=1.0, key=f"h6_{rc}")
        target_min_roi_hurdle = roi_input_val / 100
    with p7: check_accretion = st.toggle("Hurdle: Deal Must Be Margin Accretive", value=False, key=f"h7_{rc}")

    st.markdown("<div class='section-header-blue'>Step 1: Target Basic Information</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: target_name = st.text_input("Target Name", value="", key=f"tn_{rc}")
    with c2: target_state = st.selectbox("State", ["", "Minnesota", "Wisconsin", "Iowa", "Illinois", "Florida"], index=0, key=f"ts_{rc}")
    with c3: target_city = st.text_input("City", value="", key=f"tc_{rc}")
    with c4: target_closing = st.date_input("Estimated Closing Date", value=None, min_value=date.today(), key=f"td_{rc}")

    st.markdown("<div class='section-header-blue'>Step 2: Financial Assumptions</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: 
        t_rev = st.number_input("Target TTM Revenue ($)", value=None, placeholder="Enter Revenue", key=f"f1a_{rc}")
        t_ebitda = st.number_input("Target TTM EBITDA ($)", value=None, placeholder="Enter EBITDA", key=f"f1b_{rc}")
    with f2:
        retention_rate = st.slider("Retention (%)", 0, 100, 90, key=f"f2a_{rc}") / 100
        organic_growth = st.slider("Year 1 Growth (%)", -50, 100, 5, key=f"f2b_{rc}") / 100
    with f3:
        ebitda_mult = st.slider("Purchase Multiple (x)", 0.0, 20.0, 5.0, 0.1, key=f"f3a_{rc}")
        synergy_pct = st.slider("Synergy Potential (%)", -50, 100, 10, key=f"f3b_{rc}") / 100
    with f4:
        fcf_conv = st.slider("FCF Conversion (%)", 0, 100, 75, key=f"f4a_{rc}") / 100
        purchase_price = (t_ebitda * ebitda_mult) if t_ebitda else 0
        st.metric("Implied Purchase Price", f"${purchase_price:,.0f}")

    # --- 3. CALCULATION GUARD ---
    checklist = {
        "Identity": sim_first != "" and sim_last != "",
        "Target": target_name != "" and target_state != "",
        "Financials": t_ebitda is not None and ebitda_mult > 0,
        "Mandate": all(v is not None for v in [target_max_lev_hurdle, target_min_ebitda_hurdle, target_max_price_hurdle])
    }
    progress_pct = sum(checklist.values()) / len(checklist)
    bar_color = "red" if progress_pct < 0.5 else "orange" if progress_pct < 1.0 else "green"
    
    st.markdown(f"### 📋 Simulation Readiness: <span style='color:{bar_color};'>{int(progress_pct*100)}%</span>", unsafe_allow_html=True)
    st.progress(progress_pct)

    if progress_pct < 1.0:
        st.info("💡 **Simulator Guard:** Complete required fields to run analysis.")
        st.stop()

    # --- 4. ENGINE (Calculated here so variables exist for all downstream logic) ---
    st.markdown("<div class='section-header-blue'>Step 3: Financing</div>", unsafe_allow_html=True)
    fi1, fi2, fi3, fi4 = st.columns(4)
    with fi1:
        debt_pct = st.slider("Bank Loan (%)", 0, 100, 50, key=f"fi1a_{rc}") / 100
        seller_pct = st.slider("Seller Note (%)", 0, 100, 10, key=f"fi1b_{rc}") / 100
    with fi2:
        int_rate_val = st.number_input("Interest Rate (%)", value=8.5, key=f"fi2a_{rc}") / 100
        loan_term = st.slider("Term (Years)", 1, 25, 7, key=f"fi2b_{rc}")
    
    new_debt = purchase_price * debt_pct
    seller_note = purchase_price * seller_pct
    total_financing = new_debt + seller_note
    cash_equity_needed = purchase_price - total_financing
    with fi3: st.metric("Total Financing", f"${total_financing:,.0f}")
    with fi4: st.metric("Cash Equity", f"${cash_equity_needed:,.0f}")

    # Core Math Logic
    mv_ebitda_base = df_master[df_master["Year"] == 2025]["EBITDA"].sum()
    mv_rev_base = df_master[df_master["Year"] == 2025]["Revenue"].sum()
    mv_fcf_base, mv_margin_base = mv_ebitda_base * 0.75, (mv_ebitda_base / mv_rev_base) * 100
    
    adj_t_ebitda = t_ebitda * retention_rate
    synergy_val = adj_t_ebitda * synergy_pct
    growth_val = (mv_ebitda_base + adj_t_ebitda) * organic_growth
    consolidated_ebitda = (mv_ebitda_base + adj_t_ebitda) + growth_val + synergy_val
    
    pf_rev_y1 = (mv_rev_base + t_rev) * (1 + organic_growth)
    pf_margin_y1 = (consolidated_ebitda / pf_rev_y1 * 100) if pf_rev_y1 > 0 else 0
    margin_impact = pf_margin_y1 - mv_margin_base
    
    total_leverage = (total_financing + 1500000) / consolidated_ebitda
    annual_debt_svc = (new_debt / loan_term) + (new_debt * int_rate_val)
    deal_dscr = consolidated_ebitda / annual_debt_svc if annual_debt_svc > 0 else 0
    net_fcf = (consolidated_ebitda * fcf_conv) - annual_debt_svc
    cash_roi = (net_fcf / cash_equity_needed * 100) if cash_equity_needed > 0 else 0

    # Verdict Logic
    fail_reasons = []
    if total_leverage > target_max_lev_hurdle: fail_reasons.append("Leverage Gap")
    if ebitda_mult > target_max_price_hurdle: fail_reasons.append("Over Valuation")
    if deal_dscr < target_min_dscr_hurdle: fail_reasons.append("Insolvent Coverage")
    if cash_equity_needed > target_available_cash: fail_reasons.append("Insufficient Cash")
    
    if fail_reasons: verdict, v_color, action = "FAIL: DE-PRIORITIZE DEAL", "#e74c3c", "Action: Terminate Deal"
    else: verdict, v_color, action = "PASS: STRATEGIC FIT", "#27ae60", "Action: Proceed with LOI"
    reason_str = " | ".join(fail_reasons) if fail_reasons else "All hurdles cleared."

    st.markdown(f"<div style='background-color:{v_color}; padding:25px; border-radius:10px; text-align:center; color:white; margin-bottom:20px;'><h2 style='color:white; margin:0;'>{verdict}</h2><p style='color:white; font-weight:bold; margin:5px 0;'>{action}</p><p style='color:white; font-size:0.9em; margin:0;'>{reason_str}</p></div>", unsafe_allow_html=True)

    # --- 5. 8-KPI GRID ---
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("DSCR", f"{deal_dscr:.2f}x", delta=f"{deal_dscr - target_min_dscr_hurdle:.2f} vs Hurdle")
    with k2: st.metric("Combined EBITDA", f"${consolidated_ebitda:,.0f}", delta=f"${consolidated_ebitda - target_min_ebitda_hurdle:,.0f} vs Goal")
    with k3: st.metric("Net Free Cash Flow", f"${net_fcf:,.0f}", delta=f"${net_fcf - mv_fcf_base:,.0f} vs MV Base")
    with k4: st.metric("Cash ROI", f"{cash_roi:.1f}%", delta=f"{cash_roi - (target_min_roi_hurdle*100):.1f}% vs Goal")

    k5, k6, k7, k8 = st.columns(4)
    with k5: st.metric("Debt / EBITDA", f"{total_leverage:.2f}x", delta=f"{total_leverage - target_max_lev_hurdle:.2f} vs Hurdle", delta_color="inverse")
    with k6: st.metric("Margin Profile", f"{pf_margin_y1:.1f}%", delta=f"{margin_impact:+.1f}% vs Base")
    with k7: st.metric("Cash Required", f"${cash_equity_needed:,.0f}", delta=f"${target_available_cash - cash_equity_needed:,.0f} Budget")
    with k8: st.metric("EBITDA Multiple", f"{ebitda_mult:.1f}x", delta=f"{ebitda_mult - target_max_price_hurdle:.1f} vs Limit", delta_color="inverse")

    # --- 6. TABS ---
    st.divider()
    tabs_list = ["📊 Financial Bridge", "📈 Amortization", "🧪 Sensitivity Analysis", "📝 Record Management"]
    st.session_state.active_tab = st.radio("Nav", tabs_list, index=tabs_list.index(st.session_state.active_tab), horizontal=True, label_visibility="collapsed", key=f"nav_{rc}")
    st.divider()

    if st.session_state.active_tab == "📊 Financial Bridge":
        labels = ["Base", "Target", "Growth", "Synergies", "Pro-Forma"]
        y_vals = [mv_ebitda_base, adj_t_ebitda, growth_val, synergy_val, consolidated_ebitda]
        fig = go.Figure(go.Waterfall(measure=["relative"]*4 + ["total"], x=labels, y=y_vals, textposition="outside",
            increasing={"marker": {"color": "#27ae60"}}, decreasing={"marker": {"color": "#e74c3c"}}, totals={"marker": {"color": "#2980b9"}}))
        st.plotly_chart(fig, use_container_width=True)

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
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_amort.to_excel(writer, index=False, sheet_name='Amortization')
        st.download_button("📥 Export Schedule", buffer.getvalue(), f"Amortization_{target_name}.xlsx", key=f"dl_am_{rc}")

    elif st.session_state.active_tab == "🧪 Sensitivity Analysis":
        st.subheader("🧪 Strategic Risk Analysis")
        c1, c2 = st.columns([3, 1])
        with c1:
            def calc_roi_sens(m, r):
                eq = (t_ebitda * m) * (1 - debt_pct - seller_pct)
                y1 = ((mv_ebitda_base + (t_ebitda * r)) * (1 + organic_growth)) + (t_ebitda * r * synergy_pct)
                return round(((y1 * fcf_conv) - annual_debt_svc) / eq * 100, 1) if eq > 0 else 0
            r_rng, m_rng = [retention_rate+i for i in [-0.1, -0.05, 0, 0.05, 0.1]], [ebitda_mult+i for i in [1, 0.5, 0, -0.5, -1]]
            sens_roi = pd.DataFrame([[calc_roi_sens(m, r) for r in r_rng] for m in m_rng], index=[f"{m:.1f}x" for m in m_rng], columns=[f"{int(r*100)}%" for r in r_rng])
            st.plotly_chart(px.imshow(sens_roi, text_auto=True, color_continuous_scale="RdYlGn", title="ROI Sensitivity"), use_container_width=True)
        with c2: st.info("**Matrix A:** Yield safety relative to retention risk.")

    elif st.session_state.active_tab == "📝 Record Management":
        if not df_hist_master.empty:
            # Multi-field Filters
            with st.form("vault_search_form"):
                f1, f2, f3 = st.columns(3)
                with f1: v_filt = st.multiselect("Verdict", options=df_hist_master["Verdict"].unique(), default=list(df_hist_master["Verdict"].unique()))
                with f2: t_filt, city_filt = st.text_input("Target Search"), st.text_input("City Search")
                with f3: d_range = st.date_input("Date Range", value=(df_hist_master["Timestamp_DT"].min().date(), date.today()))
                if st.form_submit_button("🔍 Run Search"):
                    mask = (df_hist_master["Verdict"].isin(v_filt))
                    if t_filt: mask = mask & (df_hist_master["Target"].str.contains(t_filt, case=False))
                    if city_filt: mask = mask & (df_hist_master["City"].str.contains(city_filt, case=False))
                    st.session_state.filtered_vault = df_hist_master[mask]

            display_df = st.session_state.get("filtered_vault", df_hist_master)
            st.dataframe(display_df.drop(columns=["ROI_num", "Lev_num", "Timestamp_DT"], errors='ignore').set_index("Sim_ID"), use_container_width=True)

            # Historical Recovery with Binary Fix
            st.markdown("### 📄 Recover Historical Memo")
            sel_rec = st.selectbox("Select Record:", options=display_df.apply(lambda x: f"ID: {x['Sim_ID']} | {x['Target']}", axis=1), key=f"recovery_{rc}")
            if st.button("📑 Generate Selected Memo"):
                sid = int(sel_rec.split("|")[0].replace("ID: ", "").strip())
                row = df_hist_master[df_hist_master["Sim_ID"] == sid].iloc[0]
                try:
                    hist_data = {"Target": str(row["Target"]), "Simulator": str(row["Simulator"]), "State": str(row["State"]), "Verdict": str(row["Verdict"]), "ROI": str(row["ROI"]), "Leverage": str(row["Leverage"])}
                    pdf_bytes = create_pdf(hist_data, {"Reasons": str(row["Reasons"])})
                    st.download_button("📥 Download Recovered PDF", pdf_bytes, f"Recovered_{row['Target']}.pdf", "application/pdf", key=f"dl_hist_{sid}")
                except Exception as e: st.error(f"Error: {e}")

    # --- 7. FINAL ACTIONS ---
    st.divider()
    strat_notes = st.text_area("🗒️ Strategic Deal Memo", key=f"memo_{rc}")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📁 Save to Database", use_container_width=True, key=f"save_{rc}"):
            new_entry = {"Sim_ID": len(df_hist_master)+1, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "Simulator": full_sim_name, "Target": target_name, "City": target_city, "State": target_state, "Verdict": verdict.split(":")[0], "ROI": f"{cash_roi:.1f}%", "Leverage": f"{total_leverage:.2f}x", "Reasons": f"{reason_str} | Notes: {strat_notes}"}
            pd.concat([df_hist_master, pd.DataFrame([new_entry])]).to_csv(db_file, index=False)
            st.success("✅ Simulation Saved Successfully.")
    with b2:
        if st.button("📄 Generate PDF Summary", use_container_width=True, key=f"pdf_btn_{rc}"):
            try:
                live_pdf_data = {"Target": str(target_name), "Simulator": str(full_sim_name), "State": str(target_state), "Verdict": f"{verdict} ({action})", "ROI": f"{cash_roi:.1f}%", "Leverage": f"{total_leverage:.2f}x"}
                pdf_bytes = create_pdf(live_pdf_data, {"Purchase Price": f"${purchase_price:,.0f}", "Reasons": str(strat_notes) if strat_notes else "N/A"})
                st.download_button("📥 Download PDF", pdf_bytes, f"Summary_{target_name}.pdf", "application/pdf", key=f"dl_pdf_{rc}")
            except Exception as e: st.error(f"PDF Error: {e}")
    with b3:
        if st.button("🔄 Restart Simulation", use_container_width=True, key=f"restart_{rc}"):
            st.session_state.update({"reset_counter": rc + 1, "active_tab": "📊 Financial Bridge"})
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