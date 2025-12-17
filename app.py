import streamlit as st
import pandas as pd
import numpy as np
import datetime
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
    /* 1. GENERAL PADDING - Reduced top padding to move content up */
    .block-container {
        padding-top: 1rem; /* Reduced from 2rem to move title up */
        padding-bottom: 2rem;
    }
    
    /* 2. SIDEBAR LAYOUT STYLING */
    /* Remove default sidebar padding to move Navigation to the very top */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem; 
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    
    /* Force sidebar content to flex column and push footer to bottom */
    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        justify-content: flex-start;
    }
    
    /* Target the caption/footer to sit at the bottom */
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
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data
def load_data_final():
    dates = pd.date_range(start="2021-01-01", end="2025-12-31", freq="D")
    
    subsidiaries = [
        "All About Caring - Rush City", 
        "All About Caring - Twin Cities", 
        "Communities of Care"
    ]
    
    service_types = [
        "PCA", "Homemaking", "ICLS", "HIS", 
        "Night Supervision", "Respite", "Complex RN", "Complex LPN"
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
    return 0.0

# ==========================================
# 4. VIEW: HOME
# ==========================================
def show_home():
    # Add CSS specifically to nudge the logo down slightly
    st.markdown("""
        <style>
        [data-testid="stImage"] {
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 1. Logo at Top Left
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
        # Added Creator field here
        st.markdown(f"""
            <p style='text-align: center; color: gray;'>
                <b>Date Built:</b> {datetime.date.today().strftime('%B %d, %Y')}<br>
                <b>Creator:</b> Amanda Zheng
            </p>
            """, unsafe_allow_html=True)
        
        st.success("👋 Welcome! Use the sidebar to navigate between Historical Performance and Financial Projections.")
        with st.expander("📝 Read Me: Assumptions & Notes", expanded=True):
            st.markdown("""
            * **Data Source:** Modeled data based on 2021-2025 financial trends.
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
            year_select = st.multiselect("Select Year", options=sorted(df_master["Year"].unique()), default=[2024, 2025])
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
        "Marketing Spend": "sum", "New Clients": "sum", "Lost Clients": "sum",
        "Leads Contacted": "sum", "Avg Sales Cycle": "mean",
        "Avg Contract Duration": "mean", 
        "Employee Count": "mean", "Client Count": "mean"
    }
    
    if view_by == "Week":
        group_cols = ["Year", "Week"]
    elif view_by == "Month":
        group_cols = ["Year", "Month"]
    elif view_by == "Quarter":
        group_cols = ["Year", "Quarter"]
    else:
        group_cols = ["Year"]
        
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

    df_view["Gross Margin %"] = (df_view["Gross Margin"] / df_view["Revenue"] * 100).fillna(0)
    df_view["EBITDA %"] = (df_view["EBITDA"] / df_view["Revenue"] * 100).fillna(0)
    df_view["Net Margin %"] = (df_view["Net Income"] / df_view["Revenue"] * 100).fillna(0)
    df_view["COGS %"] = (df_view["COGS"] / df_view["Revenue"] * 100).fillna(0)
    df_view["OpEx %"] = (df_view["Expenses"] / df_view["Revenue"] * 100).fillna(0)
    
    df_view["Rev per Client"] = (df_view["Revenue"] / df_view["Client Count"]).fillna(0)
    df_view["Rev per Caregiver"] = (df_view["Revenue"] / df_view["Employee Count"]).fillna(0)
    
    df_view["Service Efficiency %"] = (df_view["Scheduled Hours"] / df_view["Qualified Hours"] * 100).fillna(0)
    df_view["Scheduling Efficiency %"] = (df_view["Contact Hours"] / df_view["Scheduled Hours"] * 100).fillna(0)
    df_view["Billing Efficiency %"] = (df_view["Billable Hours"] / df_view["Contact Hours"] * 100).fillna(0)
    
    df_view["New Employee %"] = (df_view["New Hires"] / df_view["Employee Count"] * 100).fillna(0)
    df_view["Turnover Rate %"] = (df_view["Departures"] / df_view["Employee Count"] * 100).fillna(0)
    
    df_view["Contact Hrs per Client"] = (df_view["Contact Hours"] / df_view["Client Count"]).fillna(0)
    df_view["Contact/Sched per Client %"] = df_view["Scheduling Efficiency %"]
    df_view["Contact/Qual per Client %"] = (df_view["Contact Hours"] / df_view["Qualified Hours"] * 100).fillna(0)
    
    df_view["CAC"] = (df_view["Marketing Spend"] / df_view["New Clients"]).replace([np.inf, -np.inf], 0).fillna(0)
    df_view["CAC / Rev per Client %"] = (df_view["CAC"] / df_view["Rev per Client"] * 100).fillna(0)
    
    df_view["COGS per Client"] = (df_view["COGS"] / df_view["Client Count"]).fillna(0)
    df_view["CAC / COGS per Client %"] = (df_view["CAC"] / df_view["COGS per Client"] * 100).fillna(0)
    df_view["Sales Conv %"] = (df_view["New Clients"] / df_view["Leads Contacted"] * 100).fillna(0)

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

    tabs = st.tabs(["💰 Revenue", "📈 Profitability", "📉 Cost Analysis", "⚙️ Operations", "👥 Human Resource", "🤝 Clients", "📢 Marketing & Sales Efficiency", "📥 Raw Data"])
    
    with tabs[0]: # Revenue
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_line_chart(df_view, "Period", "Revenue", "Total Revenue Trend"), use_container_width=True)
        with c2:
            if "Service Type" not in group_cols:
                grp_svc = group_cols + ["Service Type"]
                df_svc_view = df_filt.groupby(grp_svc)["Revenue"].sum().reset_index()
                
                # Apply same period logic to this specific dataframe
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
        st.subheader("Margin Analysis Over Time")
        fig_margins = go.Figure()
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['Gross Margin %'], name='Gross Margin %'))
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['EBITDA %'], name='EBITDA %'))
        fig_margins.add_trace(go.Scatter(x=df_view['Period'], y=df_view['Net Margin %'], name='Net Margin %', line=dict(dash='dot')))
        fig_margins.update_layout(template="plotly_white", hovermode="x unified")
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

    with tabs[5]: # Clients
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_line_chart(df_view, "Period", "Client Count", "Total Active Clients"), use_container_width=True)
        with c2: st.plotly_chart(plot_line_chart(df_view, "Period", "Avg Contract Duration", "Avg Contract Duration (Days)"), use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Client Flows (Inflow/Outflow)")
            fig_client_io = go.Figure()
            fig_client_io.add_trace(go.Bar(x=df_view['Period'], y=df_view['New Clients'], name='New Clients', marker_color='#2ecc71'))
            fig_client_io.add_trace(go.Bar(x=df_view['Period'], y=-df_view['Lost Clients'], name='Lost Clients', marker_color='#e74c3c'))
            fig_client_io.update_layout(barmode='relative', template="plotly_white", hovermode="x unified", yaxis_title="Count")
            st.plotly_chart(fig_client_io, use_container_width=True)
        with c4: st.plotly_chart(plot_line_chart(df_view, "Period", "Contact Hrs per Client", "Avg Contact Hrs/Client"), use_container_width=True)
        
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(plot_line_chart(df_view, "Period", "Contact/Sched per Client %", "Avg Contact/Sched Hours %"), use_container_width=True)
        with c6: st.plotly_chart(plot_line_chart(df_view, "Period", "Contact/Qual per Client %", "Avg Contact/Qualified Hours %"), use_container_width=True)

    with tabs[6]: # Marketing & Sales
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
        hist_basis = st.selectbox("", ["Monthly", "Quarterly", "Annual"])
    with c3:
        st.markdown("<div class='step-header'>Baseline Year:</div>", unsafe_allow_html=True)
        avail_years = [2025, 2024]
        basis_year = st.selectbox("", avail_years, index=0)
    with c4:
        st.markdown("<div class='step-header'>Duration:</div>", unsafe_allow_html=True)
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
    with o1: inputs['tax_rate_corp'] = st.number_input("Effective Tax Rate %", value=25.0) / 100
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
    else: steps_per_year = 1
        
    duration_map = {"1 Quarter": 0.25, "1 Year": 1, "3 Years": 3, "5 Years": 5}
    total_steps = int(duration_map[proj_years] * steps_per_year)
    if total_steps < 1: total_steps = 1

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

        for i in range(1, total_steps + 1):
            curr_patients *= (1 + p_pat_g)
            curr_rate *= (1 + p_rate_g)
            curr_wage_u *= (1 + p_wage_g)
            curr_wage_s *= (1 + p_wage_g)
            curr_opex_base *= (1 + p_exp_g)

            hrs_qual = inputs['base_qual_hours']
            hrs_sched = hrs_qual * inputs['sched_qual_pct']
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

            if hist_basis == "Monthly":
                date_val = pd.to_datetime(start_date) + pd.DateOffset(months=i)
                lbl = f"{date_val.year}-{date_val.month:02d}"
            elif hist_basis == "Quarterly":
                date_val = pd.to_datetime(start_date) + pd.DateOffset(months=i*3)
                lbl = f"{date_val.year}-Q{(date_val.month-1)//3+1}"
            else:
                date_val = pd.to_datetime(start_date) + pd.DateOffset(years=i)
                lbl = str(date_val.year)

            data_rows.append({
                "Scenario": sc, "Period": lbl, 
                "Revenue": revenue, "EBITDA": ebitda, "Net Income": net_income,
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

    with tab_data:
        pivot = df_all.pivot(index="Period", columns="Scenario", values=["Revenue", "EBITDA", "Net Income"])
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)
        csv_proj = convert_df_to_csv(df_all)
        st.download_button("📥 Download Projections CSV", data=csv_proj, file_name="MagnaVita_Projections.csv", mime="text/csv")

# ==========================================
# 7. MAIN APP NAVIGATION
# ==========================================
def main():
    # Sidebar Navigation - No heading, just the radio buttons at the top
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Historical Performance", "Financial Projections"], label_visibility="collapsed")
    
    # Sidebar Footer - Pushed to bottom by CSS
    st.sidebar.caption("v3.16 | Updated for Streamlit")
    
    if page == "Home":
        show_home()
    elif page == "Historical Performance":
        show_history()
    elif page == "Financial Projections":
        show_projections()

if __name__ == "__main__":
    main()