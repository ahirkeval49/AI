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

# ---------------------------------------------------------
# CMU BRAND COLORS & UI CONFIG
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#0f0f0f" # Softer dark background for readability
CARD_BG = "#1a1a1a"

st.set_page_config(page_title="CMU Data Systems", layout="wide", initial_sidebar_state="collapsed")

# Clean, readable Dark Mode CSS with CMU Red Accents
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 800; font-family: 'Segoe UI', sans-serif; }}
    p, span, div, label, li, td, th {{ color: {WHITE}; font-family: 'Segoe UI', sans-serif; }}
    
    .console-card {{
        background-color: {CARD_BG}; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 24px;
        border-top: 3px solid {CMU_RED}; border-left: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;
    }}
    
    div[data-testid="stMetricValue"] {{ color: {WHITE} !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] {{ background-color: #222; padding: 15px; border-radius: 10px; border-left: 4px solid {CMU_RED}; }}
    
    .nav-grid {{ display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 20px; }}
    .nav-card {{
        background: #222; border: 1px solid #444; border-radius: 8px; padding: 10px 20px;
        text-align: center; text-decoration: none; color: {WHITE} !important; font-weight: 700;
        transition: 0.2s ease-in-out; text-transform: uppercase; letter-spacing: 1px; font-size: 12px;
    }}
    .nav-card:hover {{ background: {CMU_RED}; border-color: {CMU_RED}; transform: translateY(-3px); }}
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

query_params = st.query_params.to_dict()
current_page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------------------------------------------------------
# DATA ENGINEERING: THE SPLIT-KEY PIPELINE
# ---------------------------------------------------------
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
                        return pd.read_csv(path, skiprows=skiprows) if f.lower().endswith('.csv') else pd.read_excel(path, skiprows=skiprows)
                    except Exception: pass
    return None

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

def clean_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

@st.cache_data
def build_master_pipeline():
    """The deeply corrected Split-Key relational join."""
    try:
        # --- 1. LOAD INDEX (THE ROSETTA STONE) ---
        idx_raw = smart_load('ucmcampaignindex')
        if idx_raw is None or idx_raw.empty: return pd.DataFrame()
        
        idx = pd.DataFrame()
        idx['board_key'] = normalize_key(idx_raw.get('Monday_Board_Name', pd.Series()))
        idx['plat_key'] = normalize_key(idx_raw.get('Campaign_ID', pd.Series()))
        idx['ga_key'] = normalize_key(idx_raw.get('UTM campaign', pd.Series()))
        idx['Category'] = idx_raw.get('Category', pd.Series()).fillna('Uncategorized')
        idx['Vendor'] = idx_raw.get('Vendor', pd.Series()).fillna('Unknown')
        idx['Display_Name'] = idx_raw.get('Monday_Board_Name', pd.Series())
        idx = idx.drop_duplicates(subset=['board_key']).dropna(subset=['board_key'])

        # --- 2. LOAD MONDAY.COM (BUDGETS & DATES) ---
        mon1 = smart_load('202425campaignmanagement')
        mon2 = smart_load('202526campaignmanagement')
        mon_raw = pd.concat([df for df in [mon1, mon2] if df is not None])
        if not mon_raw.empty:
            mon = pd.DataFrame()
            mon['board_key'] = normalize_key(mon_raw.get('Name', pd.Series()))
            mon['Budget'] = clean_num(mon_raw.get('Budget', pd.Series()))
            mon['Run_Dates'] = mon_raw.get('Run Dates', pd.Series()).astype(str)
            mon = mon.groupby('board_key').first().reset_index()
            idx = pd.merge(idx, mon, on='board_key', how='left')

        # --- 3. LOAD PLATFORMS (GOOGLE + LINKEDIN) ---
        plat_dfs = []
        for f in ['gadsfy25totals', 'gadsfy26totals', 'gadsfy24fy26monthlyweeklyperformance']:
            g_df = smart_load(f)
            if g_df is not None and not g_df.empty:
                camp_col = next((c for c in g_df.columns if 'campaign' in str(c).lower() or 'ad name' in str(c).lower()), None)
                if camp_col:
                    g_df = g_df[~g_df[camp_col].astype(str).str.contains('Total', case=False, na=False)]
                    ext = pd.DataFrame()
                    ext['plat_key'] = normalize_key(g_df[camp_col])
                    ext['Spend'] = clean_num(g_df.get('Cost', g_df.get('Spend', pd.Series(0))))
                    ext['Clicks'] = clean_num(g_df.get('Clicks', pd.Series(0)))
                    plat_dfs.append(ext)
        
        li_df = smart_load('linkedinadperformance')
        if li_df is not None and not li_df.empty:
            camp_col = next((c for c in li_df.columns if 'campaign' in str(c).lower()), None)
            if camp_col:
                ext = pd.DataFrame()
                ext['plat_key'] = normalize_key(li_df[camp_col])
                ext['Spend'] = clean_num(li_df.get('Total Spend', li_df.get('Spend', pd.Series(0))))
                ext['Clicks'] = clean_num(li_df.get('Clicks', pd.Series(0)))
                plat_dfs.append(ext)
                
        plat_agg = pd.concat(plat_dfs).groupby('plat_key').sum().reset_index() if plat_dfs else pd.DataFrame(columns=['plat_key', 'Spend', 'Clicks'])

        # --- 4. LOAD IMPACT (GOOGLE ANALYTICS) ---
        ga_dfs = []
        for f in ['gafy25utmtotals', 'gafy26utmtotals']:
            ga_raw = smart_load(f, skiprows=0)
            if ga_raw is not None and not ga_raw.empty:
                if 'session campaign' not in str(ga_raw.columns).lower(): ga_raw.columns = ga_raw.iloc[0]; ga_raw = ga_raw[1:]
                camp_col = next((c for c in ga_raw.columns if 'campaign' in str(c).lower()), None)
                if camp_col:
                    ext = pd.DataFrame()
                    ext['ga_key'] = normalize_key(ga_raw[camp_col])
                    ext['Users'] = clean_num(ga_raw.get('Total users', pd.Series(0)))
                    ext['Eng_Rate'] = clean_num(ga_raw.get('Engagement rate', pd.Series(0)))
                    ext['Duration'] = clean_num(ga_raw.get('Average session duration', pd.Series(0)))
                    ga_dfs.append(ext)
                    
        ga_agg = pd.concat(ga_dfs).groupby('ga_key').agg({'Users':'sum', 'Eng_Rate':'mean', 'Duration':'mean'}).reset_index() if ga_dfs else pd.DataFrame(columns=['ga_key', 'Users', 'Eng_Rate', 'Duration'])

        # --- 5. THE MASTER MERGE ---
        master = pd.merge(idx, plat_agg, on='plat_key', how='outer')
        master = pd.merge(master, ga_agg, on='ga_key', how='outer')
        
        master['Display_Name'] = master['Display_Name'].fillna(master['plat_key']).fillna(master['ga_key']).replace('', 'Unknown Campaign')
        master['Category'] = master['Category'].fillna('Uncategorized')
        master['Vendor'] = master['Vendor'].fillna('Platform/Organic')
        
        master.fillna({'Spend':0, 'Clicks':0, 'Users':0, 'Eng_Rate':0, 'Duration':0, 'Budget':0}, inplace=True)
        
        # Calculate God-Tier KPIs
        master['Dropoff_Rate'] = np.where(master['Clicks'] > 10, ((master['Clicks'] - master['Users']) / master['Clicks']).clip(0, 1), 0)
        master['CPWU'] = np.where(master['Users'] > 0, master['Spend'] / master['Users'], 0)
        master['Engaged_Mins'] = (master['Users'] * master['Duration']) / 60
        master['CPQM'] = np.where(master['Engaged_Mins'] > 0.5, master['Spend'] / master['Engaged_Mins'], 0)
        
        master = master[(master['Spend'] > 0) | (master['Users'] > 0)]
        return master
    except Exception as e: 
        st.error(f"Pipeline Error: {e}")
        return pd.DataFrame()

@st.cache_data
def load_timeseries():
    """Unpivots Wide GA arrays into proper line-chart data."""
    try:
        ts1 = smart_load('gafy25timeseries')
        ts2 = smart_load('gafy26timeseries')
        dfs = []
        for df in [ts1, ts2]:
            if df is not None and not df.empty:
                if 'session campaign' not in str(df.columns).lower(): 
                    df.columns = df.iloc[0]
                    df = df[1:]
                
                day_cols = [c for c in df.columns if 'day' in str(c).lower() and any(char.isdigit() for char in str(c))]
                
                if day_cols:
                    m = df.melt(id_vars=[c for c in df.columns if c not in day_cols], value_vars=day_cols, var_name='D', value_name='u')
                    m['Day'] = m['D'].astype(str).str.extract(r'(\d+)').astype(float)
                    m['Users'] = clean_num(m['u'])
                    dfs.append(m[['Day', 'Users']])
        if dfs:
            agg = pd.concat(dfs).groupby('Day')['Users'].sum().reset_index().sort_values('Day')
            return agg
    except Exception as e: 
        pass
    
    return pd.DataFrame()

master_df = build_master_pipeline()
ts_data = load_timeseries()

# ---------------------------------------------------------
# UI: NAVIGATION & HEADER
# ---------------------------------------------------------
nav_html = """
<div class="nav-grid">
    <a href="?page=home" class="nav-card">🌌 Nexus</a>
    <a href="?page=dashboard" class="nav-card">🖥️ Architect Dashboard</a>
    <a href="?page=analysis" class="nav-card">🧪 Strategist Deep-Dive</a>
    <a href="?page=graph" class="nav-card">🕸️ Knowledge Graph</a>
</div>
"""

if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; font-size: 60px; margin-top: 100px;'>CMU COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; letter-spacing: 2px;'>DETERMINISTIC SPLIT-KEY ENGINE</p>", unsafe_allow_html=True)

# ======================= ARCHITECT DASHBOARD =======================
elif current_page == "dashboard":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🖥️ Pipeline Dashboard</h1>", unsafe_allow_html=True)
    
    # Intuitive Filters
    st.markdown("<div class='console-card'><h3>🎯 Quick Filters</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if not master_df.empty:
        vendors = master_df['Vendor'].unique().tolist()
        categories = master_df['Category'].unique().tolist()
        with c1: sel_vend = st.multiselect("Filter by Platform/Vendor", vendors, default=vendors)
        with c2: sel_cat = st.multiselect("Filter by Department/Category", categories, default=categories)
        f_df = master_df[(master_df['Vendor'].isin(sel_vend)) & (master_df['Category'].isin(sel_cat))]
    else:
        f_df = pd.DataFrame()
    st.markdown("</div>", unsafe_allow_html=True)

    if not f_df.empty:
        # Hero KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ecosystem Spend", f"${f_df['Spend'].sum():,.0f}")
        m2.metric("Total Acquired Users", f"{f_df['Users'].sum():,.0f}")
        avg_cpa = f_df['Spend'].sum() / f_df['Users'].sum() if f_df['Users'].sum() > 0 else 0
        m3.metric("System Average CPA", f"${avg_cpa:.2f}")
        avg_drop = (f_df['Clicks'].sum() - f_df['Users'].sum()) / f_df['Clicks'].sum() if f_df['Clicks'].sum() > 0 else 0
        m4.metric("Avg Pipeline Drop-off", f"{avg_drop:.1%}")

        # ROW 1: Flight Risk & Time Series
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("<div class='console-card'><h3>⏳ Flight Risk (Pacing vs Budget)</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#aaa;'>Data Source: UCM Index & Monday.com Boards</p>", unsafe_allow_html=True)
            current_date = pd.to_datetime('2026-05-03') # Preset context date
            pacing = []
            for _, r in f_df.iterrows():
                if r['Budget'] > 0 and pd.notna(r['Run_Dates']) and '-' in str(r['Run_Dates']):
                    try:
                        start_str, end_str = str(r['Run_Dates']).split('-')
                        d_start, d_end = pd.to_datetime(start_str.strip()), pd.to_datetime(end_str.strip())
                        if d_start <= d_end:
                            pct_time = min(max((current_date - d_start).days / ((d_end - d_start).days + 1), 0), 1)
                            pct_spend = r['Spend'] / r['Budget']
                            pacing.append({
                                'Campaign': r['Display_Name'][:35] + '...',
                                'Time Elapsed': pct_time,
                                'Budget Spent': pct_spend,
                                'Pacing Delta': pct_spend - pct_time
                            })
                    except: pass
            if pacing:
                p_df = pd.DataFrame(pacing).sort_values('Pacing Delta')
                st.dataframe(p_df.style.format({'Time Elapsed': '{:.1%}', 'Budget Spent': '{:.1%}', 'Pacing Delta': '{:+.1%}'}), use_container_width=True)
            else: st.info("No active budgeted campaigns found for pacing analysis.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("<div class='console-card'><h3>📈 Temporal Pulse (Users)</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#aaa;'>Data Source: GA TimeSeries</p>", unsafe_allow_html=True)
            if not ts_data.empty:
                fig_ts = px.line(ts_data, x='Day', y='Users', color_discrete_sequence=[CMU_RED])
                fig_ts.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
                st.plotly_chart(fig_ts, use_container_width=True)
            else: st.info("TimeSeries data unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ROW 2: Sankey
        st.markdown("<div class='console-card'><h3>🌊 Traffic Attribution Waterfall</h3>", unsafe_allow_html=True)
        top = f_df[f_df['Users'] > 0].sort_values('Users', ascending=False).head(15)
        if not top.empty:
            nodes = list(top['Vendor'].unique()) + list(top['Category'].unique()) + list(top['Display_Name'].unique())
            n_map = {n: i for i, n in enumerate(nodes)}
            links = []
            for _, r in top.iterrows():
                links.append({'source': n_map[r['Vendor']], 'target': n_map[r['Category']], 'value': r['Users']})
                links.append({'source': n_map[r['Category']], 'target': n_map[r['Display_Name']], 'value': r['Users']})
            l_df = pd.DataFrame(links).groupby(['source','target']).sum().reset_index()
            fig_s = go.Figure(go.Sankey(
                node=dict(label=nodes, color=CMU_RED, pad=15, thickness=20),
                link=dict(source=l_df['source'], target=l_df['target'], value=l_df['value'], color="rgba(196,18,48,0.4)")
            ))
            fig_s.update_layout(paper_bgcolor=CARD_BG, font_color=WHITE, height=450)
            st.plotly_chart(fig_s, use_container_width=True)
        else: st.info("Insufficient user volume for waterfall.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ROW 3: Attention Grid & Department Bar
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("<div class='console-card'><h3>🌍 Attention Economy Grid</h3>", unsafe_allow_html=True)
            fig_at = px.scatter(f_df[f_df['Users']>0], x="Eng_Rate", y="Duration", size="Users", hover_name="Display_Name", color="Vendor", color_discrete_sequence=[CMU_RED, "#555", "#999"])
            fig_at.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_at, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_y:
            st.markdown("<div class='console-card'><h3>🟣 Department Allocation</h3>", unsafe_allow_html=True)
            fig_bar = px.bar(f_df.groupby('Category')['Spend'].sum().reset_index(), x='Category', y='Spend', color_discrete_sequence=[CMU_RED])
            fig_bar.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Dashboard offline. Waiting for valid data files.")

# ======================= STRATEGIST DEEP-DIVE =======================
elif current_page == "analysis":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🧪 Quantitative Strategist</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='console-card'><h3>📉 Diminishing Returns (Ad Fatigue)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; color:#aaa;'>Data Source: Master Joined Pipeline (Spend vs GA Users)</p>", unsafe_allow_html=True)
        reg_df = master_df[(master_df['Spend'] > 0) & (master_df['Users'] > 0)]
        if len(reg_df) > 5:
            p = np.polyfit(reg_df['Spend'], reg_df['Users'], 2)
            f = np.poly1d(p)
            x_ax = np.linspace(reg_df['Spend'].min(), reg_df['Spend'].max(), 100)
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=reg_df['Spend'], y=reg_df['Users'], mode='markers', name='Campaigns', marker=dict(color=CMU_RED, size=8), text=reg_df['Display_Name']))
            fig_p.add_trace(go.Scatter(x=x_ax, y=f(x_ax), mode='lines', name='Fatigue Curve', line=dict(color=CMU_GREY, dash='dash', width=3)))
            fig_p.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_p, use_container_width=True)
        else: st.info("Not enough spend variance to plot regression curve.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='console-card'><h3>🏆 True Quality: Cost Per Quality Minute</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; color:#aaa;'>Data Source: Master Joined Pipeline (Spend / Total Engaged Minutes)</p>", unsafe_allow_html=True)
        cpqm_df = master_df[master_df['CPQM'] > 0].sort_values('CPQM').head(8)
        if not cpqm_df.empty:
            st.dataframe(cpqm_df[['Display_Name', 'Vendor', 'CPQM', 'Spend', 'Users']].style.format({'CPQM': '${:.2f}', 'Spend': '${:,.0f}'}), use_container_width=True)
        else: st.info("Not enough duration data to model CPQM.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='console-card'><h3>👥 Audience Segment Efficiency</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; color:#aaa;'>Data Source: Explicitly extracted from <code>GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv</code></p>", unsafe_allow_html=True)
    
    def find_col(df, aliases):
        if df is None or df.empty: return None
        for alias in aliases:
            for col in df.columns:
                if alias.lower() in str(col).lower(): return col
        return None
        
    aud_raw = smart_load('gadsaudienceperformancebycampaignfy24fy26')
    if aud_raw is not None:
        seg_col = find_col(aud_raw, ['audience segment'])
        clk_col = find_col(aud_raw, ['clicks'])
        imp_col = find_col(aud_raw, ['impr', 'impressions'])
        if seg_col and clk_col and imp_col:
            aud_raw[clk_col] = clean_num(aud_raw[clk_col])
            aud_raw[imp_col] = clean_num(aud_raw[imp_col])
            aud_agg = aud_raw.groupby(seg_col).agg({clk_col: 'sum', imp_col: 'sum'}).reset_index()
            aud_agg = aud_agg[aud_agg[imp_col] > 5000].copy() # Statistical significance threshold
            aud_agg['Click_Through_Rate'] = aud_agg[clk_col] / aud_agg[imp_col]
            top_aud = aud_agg.sort_values('Click_Through_Rate', ascending=False).head(12)
            st.dataframe(top_aud.style.format({'Click_Through_Rate': '{:.2%}', clk_col: '{:,.0f}', imp_col: '{:,.0f}'}), use_container_width=True)
        else: st.info("Audience columns not found.")
    else: st.info("Audience Performance raw file not found in root.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_html, unsafe_allow_html=True)
    st.markdown("<h1>🕸️ Ecosystem Relational Graph</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="console-card">
        <p>This physics-based network maps the successful relational joins generated by our Split-Key pipeline: <strong>Vendor → Campaign Category → Campaign Name</strong>. Nodes are draggable.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        net = Network(height="700px", width="100%", bgcolor=CARD_BG, font_color=WHITE)
        net.add_node("CMU", size=50, color=CMU_RED, label="CMU Hub")
        
        for vend in master_df['Vendor'].unique():
            if str(vend) not in ['0', 'nan', 'Unknown', 'Platform/Organic']:
                net.add_node(vend, size=35, color="#666", label=vend)
                net.add_edge("CMU", vend)
                
                vend_df = master_df[master_df['Vendor'] == vend]
                for cat in vend_df['Category'].unique():
                    cat_node = f"{vend}_{cat}"
                    net.add_node(cat_node, size=25, color="#888", label=cat)
                    net.add_edge(vend, cat_node)
                    
                    camps = vend_df[vend_df['Category'] == cat].sort_values('Users', ascending=False).head(8)
                    for _, r in camps.iterrows():
                        camp_name = str(r['Display_Name'])[:30] + "..." if len(str(r['Display_Name'])) > 30 else str(r['Display_Name'])
                        if camp_name and camp_name != "nan":
                            net.add_node(camp_name, size=15, color=CMU_RED, label=camp_name)
                            net.add_edge(cat_node, camp_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                # Inject a small CSS fix to remove the default white border from pyvis
                html_code = f.read().replace('<style type="text/css">', '<style type="text/css">\n #mynetwork {border: none; outline: none;}\n')
                components.html(html_code, height=750)
    else:
        st.error("No joined data available to render graph.")
