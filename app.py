import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
import re
from itertools import combinations

# ---------------------------------------------------------
# CMU BRAND COLORS & CONFIGURATION
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#050505"

st.set_page_config(page_title="CMU Data Systems | Command Center", layout="wide", initial_sidebar_state="expanded")

# GLOBAL RED TEXT OVERRIDE
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; }}
    .stSidebar {{ background-color: #111111; }}
    
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6, p, span, div, label, li, strong, a, th, td {{
        color: {CMU_RED} !important;
        font-family: 'Segoe UI', sans-serif;
    }}
    
    h1, h2, h3, h4, h5, h6 {{ font-weight: 900; letter-spacing: -0.5px; }}
    
    .console-card {{
        background-color: {WHITE};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(196, 18, 48, 0.3);
        margin-bottom: 24px;
        border-top: 4px solid {CMU_RED};
    }}
    
    div[data-testid="stPlotlyChart"] {{ background-color: {WHITE}; border-radius: 12px; padding: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
    div[data-testid="stMetricValue"] {{ font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }}
    div[data-testid="stMetric"] {{ background-color: {WHITE}; padding: 15px; border-radius: 12px; border-left: 5px solid {CMU_RED}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    
    .nav-grid {{ display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; z-index: 100; position: relative; }}
    .nav-card {{
        background: rgba(255, 255, 255, 0.05); border: 1px solid {CMU_RED}; 
        backdrop-filter: blur(10px); border-radius: 8px; padding: 10px;
        width: 140px; text-align: center; text-decoration: none;
        transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    .nav-card:hover {{ background: rgba(196, 18, 48, 0.2); transform: translateY(-5px); box-shadow: 0 8px 20px rgba(196,18,48,0.8); }}
    .nav-title {{ font-size: 10px; font-weight: 900; color: {WHITE}; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; }}
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

query_params = st.query_params.to_dict()
current_page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------------------------------------------------------
# OMNI-LOADER & ETL PIPELINE (Deterministic Logic)
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM_Campaign_Index.csv"
]

def sanitize_filename(name):
    return re.sub(r'[^a-z0-9]', '', name.lower().replace('.csv', '').replace('.xlsx', ''))

@st.cache_data(ttl=60)
def smart_load(target_name, skiprows=0):
    sanitized_target = sanitize_filename(target_name)
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), os.path.join(os.getcwd(), 'data')]
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for f in os.listdir(d):
                if sanitize_filename(f) == sanitized_target or sanitize_filename(f).startswith(sanitized_target):
                    path = os.path.join(d, f)
                    try:
                        if f.lower().endswith('.csv'): return pd.read_csv(path, skiprows=skiprows)
                        elif f.lower().endswith(('.xls', '.xlsx')): return pd.read_excel(path, skiprows=skiprows)
                    except Exception: pass
    return None

def find_col(df, aliases):
    if df is None or df.empty: return None
    for alias in aliases:
        for col in df.columns:
            if alias.lower() in str(col).lower(): return col
    return None

def clean_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

# ---------------------------------------------------------
# ALCHEMIST: THE MASTER CROSS-REFERENCE HUB
# ---------------------------------------------------------
@st.cache_data
def build_master_hub():
    """Builds a master relational spine from all execution layers."""
    try:
        # 1. Load Rosetta Stone (Campaign Index)
        idx = smart_load('ucmcampaignindex')
        if idx is not None and not idx.empty:
            utm_col = find_col(idx, ['utm campaign', 'campaign_id', 'utm_combined_id'])
            idx['utm_clean'] = normalize_key(idx[utm_col]) if utm_col else ""
            cat_col = find_col(idx, ['tactic/category', 'category'])
            idx['Category'] = idx[cat_col].fillna("Uncategorized") if cat_col else "Uncategorized"
            bud_col = find_col(idx, ['budget'])
            idx['Raw_Budget'] = clean_num(idx[bud_col]) if bud_col else 0.0
            idx = idx[['utm_clean', 'Category', 'Raw_Budget']].drop_duplicates(subset=['utm_clean'])
        else:
            idx = pd.DataFrame(columns=['utm_clean', 'Category', 'Raw_Budget'])

        # 2. Execution Layer: Platform Spend (Google & LinkedIn)
        g_dfs = []
        for f in ['gadsfy25totals', 'gadsfy26totals', 'gadsfy24fy26monthlyweeklyperformance']:
            df = smart_load(f)
            if df is not None and not df.empty:
                g_key = find_col(df, ['ad name', 'campaign'])
                if g_key:
                    df['utm_clean'] = normalize_key(df[g_key])
                    cost_col = find_col(df, ['cost', 'spend'])
                    clk_col = find_col(df, ['clicks'])
                    if cost_col: df['Cost'] = clean_num(df[cost_col])
                    if clk_col: df['Clicks'] = clean_num(df[clk_col])
                    extract = ['utm_clean']
                    if cost_col: extract.append('Cost')
                    if clk_col: extract.append('Clicks')
                    g_dfs.append(df[extract])

        g_agg = pd.concat(g_dfs, ignore_index=True).groupby('utm_clean').agg(
            GAds_Spend=('Cost', 'sum') if 'Cost' in pd.concat(g_dfs).columns else None,
            GAds_Clicks=('Clicks', 'sum') if 'Clicks' in pd.concat(g_dfs).columns else None
        ).reset_index() if g_dfs else pd.DataFrame(columns=['utm_clean', 'GAds_Spend', 'GAds_Clicks'])

        li = smart_load('linkedinadperformance')
        li_agg = pd.DataFrame(columns=['utm_clean', 'LI_Spend', 'LI_Clicks'])
        if li is not None and not li.empty:
            li_key = find_col(li, ['campaign name', 'campaign'])
            if li_key:
                li['utm_clean'] = normalize_key(li[li_key])
                li['LI_Spend'] = clean_num(li[find_col(li, ['total spend', 'spend', 'cost'])])
                li['LI_Clicks'] = clean_num(li[find_col(li, ['clicks'])])
                li_agg = li.groupby('utm_clean').agg(LI_Spend=('LI_Spend', 'sum'), LI_Clicks=('LI_Clicks', 'sum')).reset_index()

        # 3. Impact Layer: Website Results (GA)
        ga_dfs = []
        for f in ['gafy25utmtotals', 'gafy26utmtotals']:
            _df = smart_load(f, skiprows=0) 
            if _df is not None and not _df.empty:
                if 'session campaign' not in str(_df.columns).lower() and len(_df) > 1:
                    _df.columns = [str(c) for c in _df.iloc[0]] 
                    _df = _df[1:]
                ga_key = find_col(_df, ['session campaign', 'campaign'])
                if ga_key:
                    _df['utm_clean'] = normalize_key(_df[ga_key])
                    _df['Total_Users'] = clean_num(_df[find_col(_df, ['total users', 'users'])])
                    _df['Engagement_Rate'] = clean_num(_df[find_col(_df, ['engagement rate'])])
                    _df['Session_Duration'] = clean_num(_df[find_col(_df, ['average session duration', 'duration'])])
                    ga_dfs.append(_df[['utm_clean', 'Total_Users', 'Engagement_Rate', 'Session_Duration']])
                    
        ga_agg = pd.concat(ga_dfs, ignore_index=True).groupby('utm_clean').agg(
            Total_Users=('Total_Users', 'sum'), 
            Engagement_Rate=('Engagement_Rate', 'mean'), 
            Session_Duration=('Session_Duration', 'mean')
        ).reset_index() if ga_dfs else pd.DataFrame(columns=['utm_clean', 'Total_Users', 'Engagement_Rate', 'Session_Duration'])

        # 4. Final Relational Join (Outer to capture Orphan Spend)
        hub = pd.merge(ga_agg, g_agg, on='utm_clean', how='outer')
        hub = pd.merge(hub, li_agg, on='utm_clean', how='outer')
        hub = pd.merge(hub, idx, on='utm_clean', how='left')
        
        hub.fillna({'GAds_Spend':0, 'LI_Spend':0, 'GAds_Clicks':0, 'LI_Clicks':0, 'Total_Users':0, 'Session_Duration':0, 'Category':'Uncategorized'}, inplace=True)
        hub['Total_Spend'] = hub['GAds_Spend'] + hub['LI_Spend']
        hub['Total_Clicks'] = hub['GAds_Clicks'] + hub['LI_Clicks']
        hub['Vendor'] = np.where(hub['GAds_Spend'] > 0, 'Google Ads', np.where(hub['LI_Spend'] > 0, 'LinkedIn', 'Organic/Unknown'))
        
        # HERO KPI: Cost Per Quality Minute (CPQM)
        hub['Total_Engaged_Minutes'] = (hub['Total_Users'] * hub['Session_Duration']) / 60
        hub['CPQM'] = np.where(hub['Total_Engaged_Minutes'] > 0.5, hub['Total_Spend'] / hub['Total_Engaged_Minutes'], 0.0)
        
        # HERO KPI: Drop-off Rate (Efficiency Leak)
        hub['Dropoff_Rate'] = np.where(hub['Total_Clicks'] > 10, ((hub['Total_Clicks'] - hub['Total_Users']) / hub['Total_Clicks']).clip(0,1), 0.0)
        
        return hub[hub['utm_clean'] != 'nan']
    except Exception: return pd.DataFrame()

@st.cache_data
def load_timeseries_data():
    """Unpivots GA Wide-Format files to construct a linear timeline."""
    try:
        ts_dfs = []
        for f_name in ['gafy25timeseries', 'gafy26timeseries']:
            df = smart_load(f_name)
            if df is not None:
                day_cols = [c for c in df.columns if 'day' in str(c).lower() and any(char.isdigit() for char in str(c))]
                if day_cols:
                    melted = df.melt(id_vars=[c for c in df.columns if c not in day_cols], value_vars=day_cols, var_name='D', value_name='u')
                    melted['day_num'] = melted['D'].str.extract(r'(\d+)').astype(float)
                    melted['users'] = clean_num(melted['u'])
                    ts_dfs.append(melted[['day_num', 'users']])
        if ts_dfs:
            agg = pd.concat(ts_dfs).groupby('day_num')['users'].sum().reset_index().sort_values('day_num')
            agg['day'] = 'Day ' + agg['day_num'].astype(int).astype(str)
            return agg
    except: pass
    return pd.DataFrame()

master_df = build_master_hub()
ts_data = load_timeseries_data()

# ---------------------------------------------------------
# UI: NAVIGATION & RED THEME
# ---------------------------------------------------------
nav_html = """
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card">🌌<div class="nav-title">NEXUS</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card">🖥️<div class="nav-title">ARCHITECT</div></a>
    <a href="?page=analysis" target="_self" class="nav-card">🧪<div class="nav-title">STRATEGIST</div></a>
    <a href="?page=graph" target="_self" class="nav-card">🕸️<div class="nav-title">KNOWLEDGE</div></a>
</div>
"""

if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {CMU_RED}; margin-top: 150px; font-size: 60px;'>CMU DATA SYSTEMS</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {WHITE}; letter-spacing: 2px;'>DETERMINISTIC CROSS-REFERENCE ENGINE</p>", unsafe_allow_html=True)

elif current_page == "dashboard":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🖥️ Visual Architect: Pipeline Status</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='console-card'><h3>🎯 Ecosystem Filters</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: vendors = st.multiselect("Filter Platform", master_df['Vendor'].unique(), default=master_df['Vendor'].unique())
    with c2: categories = st.multiselect("Filter Category", master_df['Category'].unique(), default=master_df['Category'].unique())
    st.markdown("</div>", unsafe_allow_html=True)

    f_df = master_df[(master_df['Vendor'].isin(vendors)) & (master_df['Category'].isin(categories))]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracked Spend", f"${f_df['Total_Spend'].sum():,.0f}")
    m2.metric("Acquired Users", f"{f_df['Total_Users'].sum():,.0f}")
    m3.metric("Avg Drop-off", f"{(f_df['Total_Clicks'].sum() - f_df['Total_Users'].sum()) / f_df['Total_Clicks'].sum():.1%}" if f_df['Total_Clicks'].sum() > 0 else "0%")
    m4.metric("Avg CPQM", f"${f_df[f_df['CPQM']>0]['CPQM'].mean():.2f}")

    st.markdown("<div class='console-card'><h3>🌊 Master Attribution Waterfall</h3>", unsafe_allow_html=True)
    top = f_df.sort_values('Total_Users', ascending=False).head(15)
    nodes = list(top['Vendor'].unique()) + list(top['Category'].unique()) + list(top['utm_clean'].unique())
    node_map = {n: i for i, n in enumerate(nodes)}
    links = []
    for _, r in top.iterrows():
        links.append({'source': node_map[r['Vendor']], 'target': node_map[r['Category']], 'value': r['Total_Users']})
        links.append({'source': node_map[r['Category']], 'target': node_map[r['utm_clean']], 'value': r['Total_Users']})
    df_l = pd.DataFrame(links).groupby(['source','target']).sum().reset_index()
    fig_s = go.Figure(go.Sankey(node=dict(label=nodes, color=CMU_RED), link=dict(source=df_l['source'], target=df_l['target'], value=df_l['value'], color="rgba(196,18,48,0.2)")))
    fig_s.update_layout(paper_bgcolor=WHITE, font_color=CMU_RED, height=500)
    st.plotly_chart(fig_s, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='console-card'><h3>📈 Temporal Pulse</h3>", unsafe_allow_html=True)
        if not ts_data.empty:
            fig_ts = px.line(ts_data, x='day', y='users', color_discrete_sequence=[CMU_RED])
            fig_ts.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=CMU_RED)
            st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c_right:
        st.markdown("<div class='console-card'><h3>🌍 Attention Matrix</h3>", unsafe_allow_html=True)
        fig_at = px.scatter(f_df, x="Engagement_Rate", y="Session_Duration", size="Total_Users", hover_name="utm_clean", color="Vendor", color_discrete_sequence=[CMU_RED, CMU_GREY, "#000"])
        fig_at.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=CMU_RED)
        st.plotly_chart(fig_at, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif current_page == "analysis":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🧪 Quantitative Strategist</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='console-card'><h3>💡 Budget Reallocation (Pure Historical Math)</h3>", unsafe_allow_html=True)
    v = master_df[(master_df['Total_Spend'] > 500) & (master_df['Total_Users'] > 10)]
    if len(v) > 2:
        best, worst = v.loc[v['CPWU'].idxmin()], v.loc[v['CPWU'].idxmax()]
        st.success(f"Shift $5,000 from `{worst['utm_clean']}` (CPA: ${worst['CPWU']:.2f}) to `{best['utm_clean']}` (CPA: ${best['CPWU']:.2f}).")
        lift = (5000/best['CPWU']) - (5000/worst['CPWU'])
        st.info(f"Projected deterministic gain: **+{int(lift):,} users** at zero net cost.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='console-card'><h3>📉 Diminishing Returns (Polynomial Regression)</h3>", unsafe_allow_html=True)
    reg_df = master_df[(master_df['Total_Spend']>0) & (master_df['Total_Users']>0)]
    if len(reg_df) > 5:
        p = np.polyfit(reg_df['Total_Spend'], reg_df['Total_Users'], 2)
        f = np.poly1d(p)
        x_ax = np.linspace(reg_df['Total_Spend'].min(), reg_df['Total_Spend'].max(), 100)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=reg_df['Total_Spend'], y=reg_df['Total_Users'], mode='markers', name='Actual', marker=dict(color=CMU_RED)))
        fig_p.add_trace(go.Scatter(x=x_ax, y=f(x_ax), mode='lines', name='Fatigue Curve', line=dict(color=CMU_GREY, dash='dash')))
        fig_p.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=CMU_RED, xaxis_title="Spend ($)", yaxis_title="Users")
        st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif current_page == "graph":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🕸️ Knowledge Graph: Entity Relations</h1>", unsafe_allow_html=True)
    
    if not master_df.empty:
        net = Network(height="700px", width="100%", bgcolor=BLACK, font_color=CMU_RED)
        net.add_node("CMU", size=50, color=CMU_RED, label="Carnegie Mellon")
        for vend in master_df['Vendor'].unique():
            net.add_node(vend, size=30, color=CMU_GREY, label=vend)
            net.add_edge("CMU", vend)
            subset = master_df[master_df['Vendor'] == vend].sort_values('Total_Users', ascending=False).head(10)
            for _, r in subset.iterrows():
                net.add_node(r['utm_clean'], size=15, color=WHITE, label=r['utm_clean'])
                net.add_edge(vend, r['utm_clean'])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=800)
