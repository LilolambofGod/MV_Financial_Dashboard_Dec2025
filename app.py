import streamlit as st
import pandas as pd
from datetime import datetime
import io
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Magna Vita Financial Dashboard",
    page_icon="📈"
)

# Custom CSS for UI Improvements
st.markdown("""
    <style>
    /* 1. GENERAL PADDING */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
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
    now = datetime.datetime.now(user_tz)

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
                <b>Date Built:</b> {datetime.date.today().strftime('%B %d, %Y')}<br>
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
# 5. VIEW: HISTORICAL PERFORMANCE
# ==========================================
def show_history():
    st.title("🏛️ Historical Performance")
    
    with st.expander("🔎 Filter Data", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sub_select = st.multiselect("Select Subsidiary", options=df_master["Subsidiary"].unique(), default=df_master["Subsidiary"].unique())
        with c2:
            year_select = st.multiselect("Select Year", options=[2024, 2025], default=[2024, 2025])
        with c3:
            service_select = st.multiselect("Service Type", options=sorted(df_master["Service Type"].unique()), default=df_master["Service Type"].unique())
        with c4:
            view_by = st.radio("Group By", ["Week", "Month", "Quarter", "Year"], horizontal=True)

    df_filt = df_master[df_master["Subsidiary"].isin(sub_select)]
    df_filt = df_filt[df_filt["Year"].isin(year_select)]
    df_filt = df_filt[df_filt["Service Type"].isin(service_select)]
    
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
        "Avg Contract Duration": "mean", 
        "Employee Count": "mean", "Client Count": "mean"
    }
    
    if view_by == "Week": group_cols = ["Year", "Week"]
    elif view_by == "Month": group_cols = ["Year", "Month"]
    elif view_by == "Quarter": group_cols = ["Year", "Quarter"]
    else: group_cols = ["Year"]
        
    df_view = df_filt.groupby(group_cols).agg(agg_rules).reset_index()
    df_view["Net Income"] = (df_view["EBITDA"] - df_view["D&A"]) * 0.75
    
    if view_by == "Week":
        df_view["Period"] = df_view.apply(lambda x: f"{int(x['Year'])}-W{int(x['Week']):02d}", axis=1)
    elif view_by == "Month":
        df_view["Period"] = df_view.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
    elif view_by == "Quarter":
        df_view["Period"] = df_view.apply(lambda x: f"{int(x['Year'])}-Q{int(x['Quarter'])}", axis=1)
    else:
        df_view["Period"] = df_view["Year"].astype(str)

    # --- FINANCIAL RATIOS ---
    df_view["Gross Margin %"] = (df_view["Gross Margin"] / df_view["Revenue"] * 100).fillna(0)
    df_view["EBITDA %"] = (df_view["EBITDA"] / df_view["Revenue"] * 100).fillna(0)
    df_view["Net Margin %"] = (df_view["Net Income"] / df_view["Revenue"] * 100).fillna(0)
    df_view["COGS %"] = (df_view["COGS"] / df_view["Revenue"] * 100).fillna(0)
    df_view["OpEx %"] = (df_view["Expenses"] / df_view["Revenue"] * 100).fillna(0)
    
    # --- UNIT ECONOMICS ---
    df_view["Rev per Client"] = (df_view["Revenue"] / df_view["Client Count"]).fillna(0)
    df_view["Rev per Caregiver"] = (df_view["Revenue"] / df_view["Employee Count"]).fillna(0)
    
    # --- OPERATIONAL RATIOS ---
    df_view["Service Efficiency %"] = (df_view["Scheduled Hours"] / df_view["Qualified Hours"] * 100).fillna(0)
    df_view["Scheduling Efficiency %"] = (df_view["Contact Hours"] / df_view["Scheduled Hours"] * 100).fillna(0)
    df_view["Billing Efficiency %"] = (df_view["Billable Hours"] / df_view["Contact Hours"] * 100).fillna(0)
    
    # --- HR RATIOS ---
    df_view["New Employee %"] = (df_view["New Hires"] / df_view["Employee Count"] * 100).fillna(0)
    df_view["Turnover Rate %"] = (df_view["Departures"] / df_view["Employee Count"] * 100).fillna(0)
    df_view["Recruitment / Rev %"] = (df_view["Recruitment Spend"] / df_view["Revenue"] * 100).fillna(0)
    df_view["Recruitment / Rev per CG %"] = (df_view["Recruitment Spend"] / df_view["Rev per Caregiver"] * 100).fillna(0)
    
    # --- CLIENT RATIOS ---
    df_view["Contact Hrs per Client"] = (df_view["Contact Hours"] / df_view["Client Count"]).fillna(0)
    df_view["Contact/Sched per Client %"] = df_view["Scheduling Efficiency %"]
    df_view["Contact/Qual per Client %"] = (df_view["Contact Hours"] / df_view["Qualified Hours"] * 100).fillna(0)
    df_view["Caregiver to Client Ratio"] = (df_view["Employee Count"] / df_view["Client Count"]).fillna(0)
    df_view["Per Client Qual Hrs"] = (df_view["Qualified Hours"] / df_view["Client Count"]).fillna(0)
    df_view["Per Client Sched Hrs"] = (df_view["Scheduled Hours"] / df_view["Client Count"]).fillna(0)
    df_view["Per Client Contact Hrs"] = (df_view["Contact Hours"] / df_view["Client Count"]).fillna(0)
    
    # --- MARKETING RATIOS ---
    df_view["CAC"] = (df_view["Marketing Spend"] / df_view["New Clients"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_view["CAC / Rev per Client %"] = (df_view["CAC"] / df_view["Rev per Client"] * 100).fillna(0)
    df_view["COGS per Client"] = (df_view["COGS"] / df_view["Client Count"]).fillna(0)
    df_view["CAC / COGS per Client %"] = (df_view["CAC"] / df_view["COGS per Client"] * 100).fillna(0)
    df_view["Sales Conv %"] = (df_view["New Clients"] / df_view["Leads Contacted"] * 100).fillna(0)

    # --- KPI DISPLAY ---
    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${df_view['Revenue'].sum():,.0f}")
    k2.metric("Total EBITDA", f"${df_view['EBITDA'].sum():,.0f}")
    k3.metric("Total Net Income", f"${df_view['Net Income'].sum():,.0f}")
    k4.metric("Avg EBITDA Margin", f"{df_view['EBITDA %'].mean():.1f}%")
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Total Contact Hours", f"{df_view['Contact Hours'].sum():,.0f}")
    k6.metric("Number of Clients", f"{df_view['Client Count'].mean():,.0f}")
    k7.metric("Number of Caregivers", f"{df_view['Employee Count'].mean():,.0f}")
    k8.metric("Avg Net Margin", f"{df_view['Net Margin %'].mean():.1f}%")
    
    st.divider()

    tabs = st.tabs(["💰 Revenue", "📈 Profitability", "📉 Cost Analysis", "⚙️ Operations", "👥 Human Resource", "🤝 Clients", "📢 Marketing & Sales Metrics", "📥 Raw Data"])
    
    with tabs[0]: # Revenue
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_line_chart(df_view, "Period", "Revenue", "Total Revenue Trend"), use_container_width=True)
        with c2:
            if "Service Type" not in group_cols:
                grp_svc = group_cols + ["Service Type"]
                df_svc_view = df_filt.groupby(grp_svc)["Revenue"].sum().reset_index()
                
                if view_by == "Week":
                    df_svc_view["Period"] = df_svc_view.apply(lambda x: f"{int(x['Year'])}-W{int(x['Week']):02d}", axis=1)
                elif view_by == "Month":
                    df_svc_view["Period"] = df_svc_view.apply(lambda x: f"{int(x['Year'])}-{int(x['Month']):02d}", axis=1)
                elif view_by == "Quarter":
                    df_svc_view["Period"] = df_svc_view.apply(lambda x: f"{int(x['Year'])}-Q{int(x['Quarter'])}", axis=1)
                else:
                    df_svc_view["Period"] = df_svc_view["Year"].astype(str)
                
                fig_svc = px.bar(df_svc_view, x="Period", y="Revenue", color="Service Type", title="Revenue by Service Type Proportion", barmode='stack')
                fig_svc.update_layout(template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_svc, use_container_width=True)
            else:
                st.info("Group by Service Type to see breakdown.")
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Rev per Client", "Avg Revenue per Client"), use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Rev per Caregiver", "Avg Revenue per Caregiver"), use_container_width=True)

    with tabs[1]: # Profitability
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "EBITDA", "EBITDA Trend"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Net Income", "Net Income Trend"), use_container_width=True)
        
        fig_margins = go.Figure()
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['Gross Margin %'], name='Gross Margin %'))
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['EBITDA %'], name='EBITDA %'))
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['Net Margin %'], name='Net Margin %', line=dict(dash='dot')))
        fig_margins.update_layout(template="plotly_white", hovermode="x unified", title="Margin Analysis Over Time")
        st.plotly_chart(fig_margins, use_container_width=True)

    with tabs[2]: # Cost Analysis
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "COGS", "Total COGS ($)"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "COGS %", "COGS % of Revenue"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Expenses", "Total OpEx ($)"), use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "OpEx %", "OpEx % of Revenue"), use_container_width=True)

    with tabs[3]: # Operations
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Service Efficiency %", "Service Eff. (Sched/Qual %)"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Scheduling Efficiency %", "Sched Eff. (Contact/Sched %)"), use_container_width=True)
        with c3: st.plotly_chart(plot_line_chart(df_view, "Period", "Billing Efficiency %", "Billing Eff. (Billable/Contact %)"), use_container_width=True)

    with tabs[4]: # HR
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Total Employee Count")
            st.plotly_chart(plot_line_chart(df_view, "Period", "Employee Count", ""), use_container_width=True)
        with c2:
            st.subheader("HR Ratios (Inflow/Outflow)")
            fig_io = go.Figure()
            fig_io.add_trace(go.Bar(x=df_view['Period'], y=df_view['New Hires'], name='New Hires', marker_color='#2ecc71'))
            fig_io.add_trace(go.Bar(x=df_view['Period'], y=-df_view['Departures'], name='Departures', marker_color='#e74c3c'))
            fig_io.update_layout(barmode='relative', template="plotly_white", hovermode="x unified", yaxis_title="Count")
            st.plotly_chart(fig_io, use_container_width=True)
        
        st.divider()
        st.subheader("Recruitment Efficiency")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_line_chart(df_view, "Period", "Recruitment / Rev %", "Recruitment Spend % of Revenue"), use_container_width=True)
        with c4:
            st.plotly_chart(plot_line_chart(df_view, "Period", "Recruitment / Rev per CG %", "Recruitment Spend / Rev per Caregiver %"), use_container_width=True)

    with tabs[5]: # Clients
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Client Count", "Total Active Clients"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Avg Contract Duration", "Avg Contract Duration (Days)"), use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            # UPDATED: Removed subheader, put title inside chart for formatting alignment
            fig_client_io = go.Figure()
            fig_client_io.add_trace(go.Bar(x=df_view['Period'], y=df_view['New Clients'], name='New Clients', marker_color='#2ecc71'))
            fig_client_io.add_trace(go.Bar(x=df_view['Period'], y=-df_view['Lost Clients'], name='Lost Clients', marker_color='#e74c3c'))
            fig_client_io.update_layout(barmode='relative', template="plotly_white", hovermode="x unified", yaxis_title="Count", title="Client Flows (Inflow/Outflow)")
            st.plotly_chart(fig_client_io, use_container_width=True)
        with c4: 
            st.plotly_chart(plot_line_chart(df_view, "Period", "Contact Hrs per Client", "Avg Per Client Contact Hrs"), use_container_width=True)
        
        c5, c6 = st.columns(2)
        with c5: 
            st.plotly_chart(plot_line_chart(df_view, "Period", "Contact/Sched per Client %", "Avg Per Client Contact/Sched Hours %"), use_container_width=True)
        with c6: 
            st.plotly_chart(plot_line_chart(df_view, "Period", "Contact/Qual per Client %", "Avg Per Client Contact/Qualified Hours %"), use_container_width=True)

        st.divider()
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(plot_line_chart(df_view, "Period", "Caregiver to Client Ratio", "Caregiver to Client Ratio"), use_container_width=True)
            st.info("ℹ️ **Definition:** Total Employees divided by Total Clients.")
        with c8:
            # UPDATED: Removed subheader, updated chart title for clarity
            df_hours_melt = df_view.melt(id_vars=["Period"], 
                                         value_vars=["Per Client Qual Hrs", "Per Client Sched Hrs", "Per Client Contact Hrs"],
                                         var_name="Metric", value_name="Hours")
            
            fig_hours = px.bar(df_hours_melt, x="Period", y="Hours", color="Metric", barmode="group",
                               title="Avg Hours Per Client: Qualified vs. Scheduled vs. Contact")
            fig_hours.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_hours, use_container_width=True)

    with tabs[6]: # Marketing & Sales Metrics
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Total Marketing Spend")
            st.plotly_chart(plot_line_chart(df_view, "Period", "Marketing Spend", ""), use_container_width=True)
        with c2:
            st.subheader("Customer Acquisition Cost (CAC)")
            st.plotly_chart(plot_line_chart(df_view, "Period", "CAC", ""), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("CAC Efficiency %")
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(x=df_view['Period'], y=df_view['CAC / Rev per Client %'], name='CAC / Rev per Client %'))
            fig_eff.add_trace(go.Scatter(x=df_view['Period'], y=df_view['CAC / COGS per Client %'], name='CAC / COGS per Client %'))
            fig_eff.update_layout(template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_eff, use_container_width=True)
        with c4:
            st.subheader("Avg Sales Conversion %")
            st.plotly_chart(plot_line_chart(df_view, "Period", "Sales Conv %", ""), use_container_width=True)
        st.subheader("Avg Sales Cycle (Days)")
        st.plotly_chart(plot_line_chart(df_view, "Period", "Avg Sales Cycle", ""), use_container_width=True)

    with tabs[7]:
        st.dataframe(df_view.style.format({"Revenue": "${:,.0f}"}), use_container_width=True)
        st.download_button("📥 Download Data", data=convert_df_to_csv(df_view), file_name="history.csv", mime='text/csv')

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
        start_date = datetime.date.today()
        
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

    with tab_data:
        pivot = df_all.pivot(index="Period", columns="Scenario", values=["Revenue", "EBITDA", "Net Income", "Valuation"])
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)
        csv_proj = conver# ==========================================

# ==========================================
# 7. VIEW: ACQUISITION SIMULATION (FINAL FIXED)
# ==========================================
def show_acquisition():
    st.title("🤝 Strategic Acquisition Simulation")

    # --- 0. PRE-LOAD DATABASE & CLEAN DATA ---
    db_file = "acquisition_history.csv"
    db_cols = ["Sim_ID", "Timestamp", "Simulator", "Target", "State", "Verdict", "ROI", "Leverage", "Reasons"]

    # Global Defaults (Prevent NameError)
    target_name = "New Target"
    purchase_price = 0.0
    target_state = "Minnesota"
    verdict = "N/A"
    reason_str = ""
    cash_roi = 0.0
    total_leverage = 0.0

    if os.path.isfile(db_file):
        df_hist_master = pd.read_csv(db_file)
        
        # --- FIX: DATA CLEANING LAYER ---
        # 1. Fill missing columns
        for col in db_cols:
            if col not in df_hist_master.columns: df_hist_master[col] = "Unknown"
        
        # 2. Convert NaN values to "Unknown" to prevent StreamlitAPIException
        df_hist_master = df_hist_master.fillna("Unknown")
        
        # 3. Ensure numeric types are preserved after fillna
        df_hist_master["Sim_ID"] = pd.to_numeric(df_hist_master["Sim_ID"], errors='coerce').fillna(0).astype(int)
        df_hist_master["Timestamp_DT"] = pd.to_datetime(df_hist_master["Timestamp"], errors='coerce')
    else:
        df_hist_master = pd.DataFrame(columns=db_cols)

    # --- 1. SIMULATOR IDENTITY ---
    st.markdown("<div class='section-header-blue'>Simulator Identity</div>", unsafe_allow_html=True)
    id1, id2, id3 = st.columns(3)
    with id1: sim_first = st.text_input("First Name", "Amanda")
    with id2: sim_last = st.text_input("Last Name", "Zheng")
    with id3: sim_pos = st.selectbox("Position", ["CEO", "Director", "Analyst", "Consultant"])
    full_sim_name = f"{sim_first} {sim_last}"

    # --- 2. INVESTMENT POLICY ---
    st.markdown("<div class='section-header-blue'>Investment Policy: Pass/Fail Criteria</div>", unsafe_allow_html=True)
    p1, p2, p3, p_new = st.columns(4)
    with p1: target_max_lev_hurdle = st.number_input("Max Leverage Hurdle", 1.0, 6.0, 3.5, 0.1, format="%.1f")
    with p2: target_min_ebitda_hurdle = st.number_input("Min Year 1 EBITDA Hurdle ($)", 500000, 10000000, 1500000, 50000)
    with p3: target_max_price_hurdle = st.number_input("Max Multiple Hurdle (x)", 1.0, 15.0, 7.0, 0.1, format="%.1f")
    with p_new: target_available_cash = st.number_input("Available Cash for Deal ($)", 0, 50000000, 2000000, 50000)

    p4, p5, p6 = st.columns(3)
    with p4: target_min_dscr_hurdle = st.number_input("Min DSCR Hurdle", 1.0, 3.0, 1.2, 0.1, format="%.1f")
    with p5: target_min_roi_hurdle = st.number_input("Min Year 1 Cash ROI (%)", 0.0, 100.0, 15.0, 1.0, format="%.1f") / 100
    with p6: check_accretion = st.toggle("Hurdle: Deal Must Be Margin Accretive", value=True)

    # --- 3. INPUT BLOCKS (Target & Financials) ---
    st.markdown("<div class='section-header-blue'>Step 1: Target Basic Information</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: target_name = st.text_input("Target Name", "Lodges Care Provider")
    with c2: target_state = st.selectbox("State", ["Minnesota", "Wisconsin", "Iowa", "Illinois", "Florida"])
    with c3: target_city = st.text_input("City", "Minneapolis")
    with c4: target_closing = st.date_input("Estimated Closing Date", datetime.now())

    st.markdown("<div class='section-header-blue'>Step 2: Financial Assumptions</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1: 
        t_rev = st.number_input("Target TTM Revenue ($)", value=4000000)
        t_ebitda = st.number_input("Target TTM EBITDA ($)", value=600000)
    with f2:
        retention_rate = st.slider("Retention (%)", 50, 100, 95) / 100
        organic_growth = st.slider("Year 1 Growth (%)", -10, 20, 5) / 100
    with f3:
        ebitda_mult = st.slider("Purchase Multiple (x EBITDA)", 1.0, 15.0, 6.1, 0.1, format="%.1f")
        synergy_pct = st.slider("Estimated Synergies (%)", -20, 50, 15) / 100
    with f4:
        fcf_conversion = st.slider("EBITDA to FCF Conversion (%)", 40, 95, 75) / 100
        purchase_price = t_ebitda * ebitda_mult
        st.metric("Implied Purchase Price", f"${purchase_price:,.0f}")

    # --- 4. STEP 3: FINANCING ---
    st.markdown("<div class='section-header-blue'>Step 3: Financing & Cash Requirement</div>", unsafe_allow_html=True)
    fi1, fi2, fi3, fi4 = st.columns(4)
    with fi1:
        debt_pct = st.slider("Bank Loan (%)", 0, 100, 60) / 100
        seller_pct = st.slider("Seller Note (%)", 0, 100, 10) / 100
    with fi2:
        int_rate = st.number_input("Bank Interest Rate (%)", 0.0, 25.0, 8.5, 0.1, format="%.1f") / 100
        loan_term = st.slider("Loan Term (Years)", 3, 15, 7)
    with fi3:
        new_debt = purchase_price * debt_pct
        seller_note = purchase_price * seller_pct
        total_financing = new_debt + seller_note
        st.metric("Total Financing $", f"${total_financing:,.0f}")
    with fi4:
        cash_equity_needed = purchase_price - total_financing
        st.metric("Total Cash Equity Required", f"${cash_equity_needed:,.0f}")

    # --- 5. CALCULATIONS & VERDICT ---
    # Global df_master reference assumed from main app
    mv_ebitda_base = df_master[df_master["Year"] == 2025]["EBITDA"].sum()
    mv_rev_base = df_master[df_master["Year"] == 2025]["Revenue"].sum()
    mv_fcf_base = mv_ebitda_base * 0.75 
    mv_margin_base = (mv_ebitda_base / mv_rev_base) * 100
    
    adj_t_ebitda = t_ebitda * retention_rate
    base_combined = mv_ebitda_base + adj_t_ebitda
    year_1_ebitda_pre_synergy = (base_combined) * (1 + organic_growth)
    synergy_val = adj_t_ebitda * synergy_pct
    consolidated_ebitda_y1 = year_1_ebitda_pre_synergy + synergy_val
    pf_rev_y1 = (mv_rev_base + t_rev) * (1 + organic_growth)
    pf_margin_y1 = (consolidated_ebitda_y1 / pf_rev_y1) * 100
    margin_impact = pf_margin_y1 - mv_margin_base
    
    total_leverage = (total_financing + 1500000) / consolidated_ebitda_y1
    annual_debt_svc = (new_debt / loan_term) + (new_debt * int_rate)
    deal_dscr = consolidated_ebitda_y1 / annual_debt_svc if annual_debt_svc > 0 else 0
    net_fcf = (consolidated_ebitda_y1 * fcf_conversion) - annual_debt_svc
    cash_roi = (net_fcf / cash_equity_needed) * 100 if cash_equity_needed > 0 else 0

    fail_reasons = []
    if total_leverage > target_max_lev_hurdle: fail_reasons.append("Leverage Gap")
    if ebitda_mult > target_max_price_hurdle: fail_reasons.append("Over Valuation")
    if deal_dscr < target_min_dscr_hurdle: fail_reasons.append("Insolvent Coverage")
    if cash_equity_needed > target_available_cash: fail_reasons.append("Insufficient Cash")
    
    caution_reasons = []
    if (cash_roi/100) < target_min_roi_hurdle: caution_reasons.append("Low ROI")
    if consolidated_ebitda_y1 < target_min_ebitda_hurdle: caution_reasons.append("Sub-scale EBITDA")
    if check_accretion and pf_margin_y1 < mv_margin_base: caution_reasons.append("Margin Dilution")

    if fail_reasons: verdict, v_color, action = "FAIL: DE-PRIORITIZE DEAL", "#e74c3c", "Action: Terminate Deal"
    elif caution_reasons: verdict, v_color, action = "CAUTION: REVIEW REQUIRED", "#f39c12", "Action: Deep-Dive Audit Required"
    else: verdict, v_color, action = "PASS: STRATEGIC FIT", "#27ae60", "Action: Proceed"
    
    reason_str = " | ".join(fail_reasons + caution_reasons) if (fail_reasons or caution_reasons) else "All hurdles cleared."

    st.markdown(f"<div style='background-color:{v_color}; padding:25px; border-radius:10px; text-align:center;'><h2 style='color:white; margin:0;'>{verdict}</h2><p style='color:white; font-weight:bold;'>{action}</p><p style='color:white; font-size:0.85em;'>{reason_str}</p></div>", unsafe_allow_html=True)

    # --- 6. 8-KPI GRID ---
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("DSCR", f"{deal_dscr:.1f}x", delta=f"{deal_dscr - target_min_dscr_hurdle:.1f} vs Criteria")
    with k2: st.metric("Combined EBITDA", f"${consolidated_ebitda_y1:,.0f}", delta=f"${consolidated_ebitda_y1 - target_min_ebitda_hurdle:,.0f} vs Criteria")
    with k3: st.metric("Net Free Cash Flow", f"${net_fcf:,.0f}", delta=f"${net_fcf - mv_fcf_base:,.0f} vs MV Baseline")
    with k4: st.metric("Cash ROI", f"{cash_roi:.1f}%", delta=f"{cash_roi - (target_min_roi_hurdle*100):.1f}% vs Criteria")

    k5, k6, k7, k8 = st.columns(4)
    with k5: st.metric("Total Debt / EBITDA", f"{total_leverage:.2f}x", delta=f"{total_leverage - target_max_lev_hurdle:.2f} vs Criteria", delta_color="inverse")
    with k6: st.metric("Net Margin Impact", f"{pf_margin_y1:.1f}%", delta=f"{margin_impact:+.1f}% vs MV Baseline")
    with k7: st.metric("Cash Equity Needed", f"${cash_equity_needed:,.0f}", delta=f"${target_available_cash - cash_equity_needed:,.0f} vs Criteria")
    with k8: st.metric("Purchase Multiple", f"{ebitda_mult:.1f}x", delta=f"{ebitda_mult - target_max_price_hurdle:.1f} vs Criteria", delta_color="inverse")

    # --- 7. TABS ---
    t1, t2, t3, t4 = st.tabs(["📊 Financial Bridge", "📈 Amortization", "🧪 Sensitivity", "📝 Record Management"])
    
    with t1:
        
        labels = ["MV Base", "Target", "Growth", "Synergies", "Pro-Forma"]
        y_vals = [mv_ebitda_base, adj_t_ebitda, (base_combined * organic_growth), synergy_val, consolidated_ebitda_y1]
        fig = go.Figure(go.Waterfall(orientation="v", measure=["relative"]*4 + ["total"], x=labels, y=y_vals, text=[f"${v/1000:,.0f}k" for v in y_vals], textposition="outside"))
        fig.update_layout(template="plotly_white", yaxis=dict(range=[0, consolidated_ebitda_y1*1.5]))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        
        amort = []
        bal = new_debt
        for y in range(1, loan_term + 1):
            i_exp, p_pay = bal * int_rate, (new_debt / loan_term)
            bal -= p_pay
            amort.append({"Year": f"Y{y}", "Interest": i_exp, "Principal": p_pay, "Ending Balance": max(0, bal)})
        st.table(pd.DataFrame(amort).set_index("Year").style.format("${:,.0f}"))

    with t3:
        st.subheader("Leverage Sensitivity Matrix (Debt/EBITDA)")
        m_range = [ebitda_mult - 1, ebitda_mult, ebitda_mult + 1]
        r_range = [retention_rate - 0.1, retention_rate, retention_rate + 0.1]
        sens = [[((t_ebitda * m * (debt_pct + seller_pct)) + 1500000) / (mv_ebitda_base + (t_ebitda * r)) for r in r_range] for m in m_range]
        df_sens = pd.DataFrame(sens, index=[f"{m:.1f}x Mult" for m in m_range], columns=[f"{r*100:.0f}% Ret" for r in r_range])
        st.dataframe(df_sens.style.background_gradient(cmap="RdYlGn_r").format("{:.2f}x"), use_container_width=True)

    with t4:
        st.markdown("### 🔍 Advanced History Filtering")
        if not df_hist_master.empty:
            if st.button("🔄 Reset Filters"): st.rerun()
            with st.expander("Filter Controls", expanded=True):
                f_c1, f_c2, f_c3 = st.columns(3)
                with f_c1:
                    # FIX: Use unique values and handle potential empties
                    v_opts = sorted(df_hist_master["Verdict"].unique().tolist())
                    sel_verdict = st.multiselect("Verdict", options=v_opts, default=v_opts)
                    
                    s_opts = sorted(df_hist_master["State"].unique().tolist())
                    sel_state = st.multiselect("State", options=s_opts, default=s_opts)
                with f_c2:
                    sim_opts = sorted(df_hist_master["Simulator"].unique().tolist())
                    sel_sim = st.multiselect("Simulator", options=sim_opts, default=sim_opts)
                    search_target = st.text_input("Search Target")
                with f_c3:
                    # Date selection safety
                    if not df_hist_master["Timestamp_DT"].isna().all():
                        min_d, max_d = df_hist_master["Timestamp_DT"].min().date(), df_hist_master["Timestamp_DT"].max().date()
                        date_rng = st.date_input("Date Range", value=(min_d, max_d))
                    else:
                        st.write("No date data available.")
                        date_rng = []

            # Filter Implementation
            df_filt = df_hist_master.copy()
            df_filt = df_filt[(df_filt["Verdict"].isin(sel_verdict)) & (df_filt["State"].isin(sel_state)) & (df_filt["Simulator"].isin(sel_sim))]
            if search_target: df_filt = df_filt[df_filt["Target"].str.contains(search_target, case=False)]
            if len(date_rng) == 2:
                df_filt = df_filt[(df_filt["Timestamp_DT"].dt.date >= date_rng[0]) & (df_filt["Timestamp_DT"].dt.date <= date_rng[1])]

            st.dataframe(df_filt[db_cols].set_index("Sim_ID"), use_container_width=True)

            to_del = st.multiselect("Select Integer IDs to Remove", options=df_filt["Sim_ID"].astype(int).tolist())
            if to_del and st.button("🔴 Confirm Deletion & Re-index"):
                df_new = df_hist_master[~df_hist_master["Sim_ID"].isin(to_del)].copy()
                df_new["Sim_ID"] = range(1, len(df_new) + 1)
                df_new[db_cols].to_csv(db_file, index=False)
                st.rerun()
        else:
            st.info("No records found.")

    # --- 8. SAVE ACTION ---
    st.divider()
    save_key = f"save_{target_name}_{int(purchase_price)}"
    if save_key not in st.session_state:
        if st.button("🚀 Save Simulation Result"):
            next_id = 1 if df_hist_master.empty else int(df_hist_master["Sim_ID"].max() + 1)
            new_rec = {
                "Sim_ID": int(next_id), "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Simulator": full_sim_name, "Target": target_name, "State": target_state,
                "Verdict": verdict, "ROI": round(cash_roi, 1), "Leverage": round(total_leverage, 2), "Reasons": reason_str
            }
            pd.concat([df_hist_master[db_cols], pd.DataFrame([new_rec])], ignore_index=True).to_csv(db_file, index=False)
            st.session_state[save_key] = True
            st.success(f"Saved Simulation ID #{next_id}")
            st.rerun()
    else:
        st.info(f"✅ Simulation for {target_name} at ${purchase_price:,.0f} is recorded.")
        
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