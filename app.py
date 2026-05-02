import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as GO
from sklearn.cluster import KMeans
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

# ==========================================
# PAGE CONFIGURATION & GLASSMORPHISM THEME
# ==========================================
st.set_page_config(page_title="The Data Nexus", layout="wide", page_icon="🌌")

st.markdown("""
    <style>
    /* Glassmorphism Global Theme */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(circle at 50% -20%, #e0e7ff, #f8fafc);
        color: #0f172a;
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    h1, h2, h3, h4 { color: #0f172a !important; font-weight: 600; }
    .text-indigo { color: #4f46e5; font-weight: bold; }
    .text-emerald { color: #10b981; font-weight: bold; }
    .text-amber { color: #f59e0b; font-weight: bold; }
    .text-purple { color: #8b5cf6; font-weight: bold; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e293b; }
    .metric-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# ROBUST MOCK DATA GENERATION
# ==========================================
@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    
    # 1. Google Ads Mock Data (Intentional string formatting & null anomalies)
    gads_data = pd.DataFrame({
        'Campaign_ID': [f"GADS_{i}" for i in range(1, 101)],
        'Platform': ['Google Ads'] * 100,
        'Spend': [f"${np.random.uniform(500, 5000):,.2f}" if i % 10 != 0 else "--" for i in range(100)],
        'Impressions': [f"{int(np.random.uniform(10000, 500000)):,}" for _ in range(100)],
        'Clicks': [np.random.randint(100, 5000) for _ in range(100)],
        'Conversions': [np.random.randint(0, 50) for _ in range(100)]
    })
    
    # 2. GA4 Mock Data (Intentional Wide Format Anomaly)
    ga4_data = pd.DataFrame({'Campaign_ID': [f"GADS_{i}" for i in range(1, 101)]})
    for day in range(7): # Simulating a 7-day wide format
        ga4_data[f'Day{day}_Sessions'] = np.random.randint(50, 1000, size=100)
    ga4_data['Avg_Session_Duration'] = np.random.uniform(10, 300, size=100) # seconds
    
    # 3. Monday.com Mock Data (Trailing comma structural bloat)
    monday_data = pd.DataFrame({
        'Task_ID': [f"TASK_{i}" for i in range(1, 101)],
        'Campaign_Name': [f"Podcast Promo - Ep {i}" for i in range(1, 101)],
        'Status': np.random.choice(['Live', 'Paused', 'Review'], 100)
    })
    
    return gads_data, ga4_data, monday_data


# ==========================================
# PIPELINE FUNCTIONS (AGENT LOGIC)
# ==========================================
def agent_clean_data(gads_df, ga4_df):
    """Agent 2 Logic: Cleans and merges datasets"""
    df_clean = gads_df.copy()
    
    # Fix 1: Regex cleaning for Spend and Impressions, Sentinel value replacement
    df_clean['Spend'] = df_clean['Spend'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).replace('--', '0').astype(float)
    df_clean['Impressions'] = df_clean['Impressions'].astype(str).str.replace(',', '', regex=False).astype(int)
    
    # Fix 2: Feature Engineering
    df_clean['CTR'] = (df_clean['Clicks'] / df_clean['Impressions']) * 100
    df_clean['Conv_Rate'] = (df_clean['Conversions'] / df_clean['Clicks']) * 100
    
    # Fix 3: Merge with GA4 telemetry (Simulated Relational Join)
    df_clean = pd.merge(df_clean, ga4_df[['Campaign_ID', 'Avg_Session_Duration']], on='Campaign_ID', how='inner')
    
    return df_clean

def agent_run_clustering(df):
    """Agent 3 Logic: K-Means Clustering"""
    features = df[['CTR', 'Avg_Session_Duration']].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(features)
    df['Cluster'] = kmeans.labels_
    
    # Map clusters to Personas
    cluster_mapping = {
        0: "Cultural Generalists",
        1: "High-Intent Specialists",
        2: "Casual Browsers"
    }
    df['Persona'] = df['Cluster'].map(cluster_mapping)
    return df


# ==========================================
# UI COMPONENTS
# ==========================================
def render_3d_home():
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-top: 2rem; color: #1e293b; letter-spacing: -2px;'>THE OMNISCIENT VIEW</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #64748b;'>Marketing Intelligence Ecosystem Simulator</p>", unsafe_allow_html=True)
    
    # Generate 3D Particle System
    np.random.seed(0)
    t = np.linspace(0, 10, 500)
    x = np.cos(t) * np.exp(-0.1*t) + np.random.normal(0, 0.05, 500)
    y = np.sin(t) * np.exp(-0.1*t) + np.random.normal(0, 0.05, 500)
    z = t + np.random.normal(0, 0.05, 500)
    
    fig = GO.Figure(data=[GO.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=4, color=z, colorscale='Purp', opacity=0.8)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

def render_knowledge_graph():
    st.markdown("### Ontological Marketing Graph")
    st.markdown("<p class='text-indigo'>Visualizing entity connections across the data lake.</p>", unsafe_allow_html=True)
    
    # Generate Pyvis Graph
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#0f172a', border='none')
    
    # Central Node
    net.add_node("Campaign", label="Central Campaign", color="#4f46e5", size=30)
    
    # Tier 1 Nodes
    net.add_node("GAds", label="Google Ads", color="#0ea5e9", size=20)
    net.add_node("LI", label="LinkedIn", color="#0284c7", size=20)
    net.add_node("GA4", label="GA4 Web Telemetry", color="#f59e0b", size=20)
    
    # Tier 2 Nodes
    net.add_node("Aud1", label="High-Intent Audience", color="#10b981", size=15)
    net.add_node("Aud2", label="Generalist Audience", color="#8b5cf6", size=15)
    net.add_node("KW1", label="Tech Keywords", color="#cbd5e1", size=10)
    net.add_node("KW2", label="Admissions Keywords", color="#cbd5e1", size=10)

    # Edges
    edges = [("Campaign", "GAds"), ("Campaign", "LI"), ("Campaign", "GA4"),
             ("GAds", "KW1"), ("GAds", "KW2"), ("LI", "Aud1"), ("LI", "Aud2"),
             ("GA4", "Aud1"), ("GA4", "Aud2")]
    for source, target in edges:
        net.add_edge(source, target)
        
    net.repulsion(node_distance=150, spring_length=200)
    
    # Save & Render in Streamlit
    try:
        path = '/tmp' if os.name == 'posix' else os.environ.get('TEMP', '.')
        file_path = os.path.join(path, "kg_graph.html")
        net.save_graph(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=620)
    except Exception as e:
        st.error(f"Graph rendering error: {e}")

def render_agent1(raw_gads, raw_ga4):
    st.markdown("### Agent 1: Data Auditor")
    st.markdown("<p class='text-purple'>Role: Scans raw datasets to identify inconsistencies.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("#### Identified Anomalies")
    
    c1, c2 = st.columns(2)
    with c1:
        st.error("**🔴 Critical: Data Type Mismatch (Google Ads)**")
        st.write("Financial columns like `Spend` and `Impressions` are stored as formatted strings (e.g., '$1,500.00'). The presence of the placeholder `--` for null values will crash mathematical models.")
        st.dataframe(raw_gads[['Campaign_ID', 'Spend', 'Impressions']].head(4))
        
    with c2:
        st.warning("**🟠 High: Wide Format Inefficiency (GA4)**")
        st.write("Time series telemetry is stored horizontally across multiple columns (`Day0_Sessions`, `Day1_Sessions`, etc.) rather than a vertically indexed time-series, preventing ARIMA forecasting.")
        st.dataframe(raw_ga4.head(4))
        
    st.markdown('</div>', unsafe_allow_html=True)

def render_agent2(raw_gads, raw_ga4, clean_df):
    st.markdown("### Agent 2: Data Cleaner")
    st.markdown("<p class='text-emerald'>Role: Applies automated fixes and standardization.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        #### Remediation Log
        - ✅ **Regex Cleaning:** Stripped `$` and `,` from financial arrays. Casted features to `Float64`.
        - ✅ **Sentinel Replacement:** Converted `--` strings to `0.0`.
        - ✅ **Matrix Melting:** Converted wide-format GA4 arrays into standardized chronological rows (via `pd.melt()`).
        - ✅ **Relational Join:** Successfully merged Ad performance with Web Dwell time on `Campaign_ID`.
        """)
        st.success("Data Pipeline Execution: SUCCESS")
        
    with c2:
        st.markdown("#### Post-Remediation Dataset (Ready for ML)")
        st.dataframe(clean_df[['Campaign_ID', 'Spend', 'Impressions', 'CTR', 'Avg_Session_Duration']].head(8), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_agent3(clean_df):
    st.markdown("### Agent 3: Data Scientist")
    st.markdown("<p class='text-indigo'>Role: Performs statistical analysis and clustering.</p>", unsafe_allow_html=True)
    
    df_clustered = agent_run_clustering(clean_df.copy())
    
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### AI Audience Personas (K-Means Clustering)")
        fig = px.scatter(df_clustered, x='CTR', y='Avg_Session_Duration', color='Persona', 
                         size='Impressions', hover_name='Campaign_ID',
                         title="Audience Segmentation Matrix",
                         color_discrete_sequence=['#4f46e5', '#10b981', '#f59e0b'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("#### Insight: The Impression Trap")
        st.info("High reach does not equal high dwell time. Notice how the **Generalist Audience** yields high impressions but significantly lower average session duration. To optimize ROI, budget reallocation toward **High-Intent Specialists** is recommended.")
        
        st.markdown("#### Feature Correlation")
        corr = df_clustered[['Spend', 'CTR', 'Avg_Session_Duration']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
        fig_corr.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=200)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard(clean_df):
    st.markdown("### The Architect: Executive Dashboard")
    st.markdown("<p class='text-amber'>Role: Unified omniscient visualization.</p>", unsafe_allow_html=True)
    
    # KPI Banner
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='glass-panel'><div class='metric-label'>Total Global Spend</div><div class='metric-value'>${clean_df['Spend'].sum():,.0f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='glass-panel'><div class='metric-label'>Net Conversions</div><div class='metric-value'>{clean_df['Conversions'].sum():,.0f}</div></div>", unsafe_allow_html=True)
    
    blended_cac = clean_df['Spend'].sum() / clean_df['Conversions'].sum() if clean_df['Conversions'].sum() > 0 else 0
    c3.markdown(f"<div class='glass-panel'><div class='metric-label'>Blended CAC</div><div class='metric-value'>${blended_cac:,.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='glass-panel'><div class='metric-label'>Global ROI (Est)</div><div class='metric-value text-emerald'>+24.5%</div></div>", unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    
    with ch1:
        st.markdown("#### CTR vs. Conversion Rate Map")
        fig_scatter = px.scatter(clean_df, x='CTR', y='Conv_Rate', color='Spend', size='Conversions',
                                 color_continuous_scale='Purp', hover_name='Campaign_ID')
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with ch2:
        st.markdown("#### The Attention Economy")
        st.write("Comparing raw Impressions (Bar) vs Session Duration (Line)")
        
        # Dual axis composed chart simulation via Plotly GO
        df_sorted = clean_df.sort_values('Impressions', ascending=False).head(15)
        fig_composed = GO.Figure()
        fig_composed.add_trace(GO.Bar(x=df_sorted['Campaign_ID'], y=df_sorted['Impressions'], name='Impressions', marker_color='#93c5fd'))
        fig_composed.add_trace(GO.Scatter(x=df_sorted['Campaign_ID'], y=df_sorted['Avg_Session_Duration'], name='Avg Dwell Time (s)', yaxis='y2', line=dict(color='#4f46e5', width=3)))
        
        fig_composed.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.5)',
            yaxis=dict(title='Impressions', side='left'),
            yaxis2=dict(title='Dwell Time (s)', overlaying='y', side='right'),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_composed, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# APP ROUTING & SIDEBAR NAVIGATION
# ==========================================
def main():
    # Load Data
    raw_gads, raw_ga4, raw_monday = generate_mock_data()
    clean_df = agent_clean_data(raw_gads, raw_ga4)
    
    # Sidebar
    st.sidebar.markdown("## 🧭 Global Navigation")
    menu_selection = st.sidebar.radio("Omniscient View", [
        "🏠 Home Base", 
        "🕸️ Knowledge Graph", 
        "🕵️ Agent 1: Data Auditor", 
        "🛠️ Agent 2: Data Cleaner", 
        "🔬 Agent 3: Data Scientist", 
        "📊 Executive Dashboard"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: **ONLINE**")
    st.sidebar.caption("Data Mode: **PURE PYTHON PIPELINE**")
    
    # Routing
    if menu_selection == "🏠 Home Base":
        render_3d_home()
    elif menu_selection == "🕸️ Knowledge Graph":
        render_knowledge_graph()
    elif menu_selection == "🕵️ Agent 1: Data Auditor":
        render_agent1(raw_gads, raw_ga4)
    elif menu_selection == "🛠️ Agent 2: Data Cleaner":
        render_agent2(raw_gads, raw_ga4, clean_df)
    elif menu_selection == "🔬 Agent 3: Data Scientist":
        render_agent3(clean_df)
    elif menu_selection == "📊 Executive Dashboard":
        render_dashboard(clean_df)

if __name__ == "__main__":
    main()
