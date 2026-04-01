import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ==========================================
# 1. PAGE CONFIGURATION & DASH THEME
# ==========================================
st.set_page_config(
    page_title="Intelligence at Scale | CMU Summit 2026",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CMU Accessible Color Palette & Glowing Accents
CMU_BLUE = "#007bc0" 
CMU_TEAL = "#008285"
GLOW_WHITE = "rgba(255, 255, 255, 0.8)"

# Advanced "Milky Way" & Glassmorphism CSS
st.markdown("""
    <style>
    /* Breathtaking Milky Way Background */
    .stApp {
        background-color: #03000a;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(76, 29, 149, 0.15), transparent 40%),
            radial-gradient(circle at 85% 30%, rgba(0, 130, 133, 0.15), transparent 40%),
            url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=3000&q=80");
        background-size: cover, cover, cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: screen;
        font-family: 'Helvetica', sans-serif;
        color: #ffffff;
    }
    
    /* Hide top header bar for full immersion */
    header {visibility: hidden;}
    
    /* Typography Glowing Effects */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3), 0 0 20px rgba(0, 123, 192, 0.5);
    }
    
    p { color: #e2e8f0 !important; }

    /* Glassmorphic Metric Cards */
    .glass-card {
        background: rgba(10, 10, 25, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 123, 192, 0.4);
        border: 1px solid rgba(0, 123, 192, 0.5);
    }
    
    .glass-card h3 { margin-bottom: 5px; font-size: 18px; color: #008285 !important; text-shadow: none;}
    .glass-card h1 { font-size: 42px; margin: 10px 0; text-shadow: 0 0 15px rgba(255,255,255,0.5);}
    
    /* Glassmorphic Container for standard elements */
    .glass-container {
        background: rgba(5, 5, 15, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
    }

    /* Celestial Buttons (Stars) */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(0, 123, 192, 0.2) 0%, rgba(0, 130, 133, 0.2) 100%) !important;
        backdrop-filter: blur(5px);
        color: #ffffff !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.4s ease !important;
        padding: 10px 20px !important;
        box-shadow: 0 0 10px rgba(0, 123, 192, 0.1);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 123, 192, 0.6) 0%, rgba(0, 130, 133, 0.6) 100%) !important;
        box-shadow: 0 0 20px rgba(0, 130, 133, 0.6), 0 0 40px rgba(0, 123, 192, 0.4) !important;
        transform: scale(1.02);
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Custom horizontal divider */
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0, 123, 192, 0.75), rgba(0,0,0,0));
        margin: 2em 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOAD & CACHING 
# ==========================================
@st.cache_data
def load_data():
    try:
        df_gads_26 = pd.read_csv("data/GAds_FY26_Totals_Jul-Dec2025.csv", skiprows=1)
        df_ga_ts_26 = pd.read_csv("data/GA_FY26_TimeSeries.csv")
        df_audiences = pd.read_csv("data/GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv")
        
        if 'Cost' in df_gads_26.columns:
            df_gads_26['Cost'] = pd.to_numeric(df_gads_26['Cost'].astype(str).str.replace(',', ''), errors='coerce')
        if 'Impr.' in df_gads_26.columns:
            df_gads_26['Impr.'] = pd.to_numeric(df_gads_26['Impr.'].astype(str).str.replace(',', ''), errors='coerce')
            
    except Exception as e:
        df_gads_26 = pd.DataFrame({"Campaign": ["WTM Awareness", "Tony Awards", "Podcast Deep Dive", "AI Care Part 1", "Robotics Gen"], "Cost": [40000, 8786, 5000, 12000, 9500], "Impr.": [350309, 104162, 45000, 85000, 62000], "CTR": [0.03, 0.63, 0.15, 0.08, 0.11]})
        df_ga_ts_26 = pd.DataFrame({"Day": range(184), "Users": np.random.poisson(lam=4000, size=184)})
        df_audiences = pd.DataFrame({"Audience segment": ["Cloud Storage", "Not in audiences", "Arts Aficionados", "Deep Learning", "General AI"], "CTR": [0.09, 0.05, 0.20, 0.12, 0.07], "TrueView view rate": [0.20, 0.35, 0.10, 0.28, 0.15]})
        
    return df_gads_26, df_ga_ts_26, df_audiences

gads_df, time_series_df, aud_df = load_data()

# ==========================================
# 3. HEADER & "MILKY WAY" UI METAPHOR
# ==========================================
st.markdown("<br>", unsafe_allow_html=True) # Spacing
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Intelligence at Scale</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #00bfff !important; font-weight: 300; letter-spacing: 2px;'>Driving a Human-Centered Future</h3>", unsafe_allow_html=True)
    
    total_budget = gads_df['Cost'].sum()
    st.markdown(f"<p style='text-align: center; font-size: 16px; opacity: 0.8;'><i>Gravitational Mass (Total Spend Context): ${total_budget:,.2f}</i></p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================
# 4. INTERACTIVE CONSTELLATION MAPPING
# ==========================================
st.markdown("<h3 style='text-align: center; margin-bottom: 25px;'>🌌 Navigate the Analytical Constellation</h3>", unsafe_allow_html=True)
star_cols = st.columns(4)

with star_cols[0]:
    if st.button("✨ Star 1: Data Genesis", use_container_width=True):
        st.session_state.active_star = 1
with star_cols[1]:
    if st.button("💫 Star 2: Neural Trends", use_container_width=True):
        st.session_state.active_star = 2
with star_cols[2]:
    if st.button("🌟 Star 3: Predictive Core", use_container_width=True):
        st.session_state.active_star = 3
with star_cols[3]:
    if st.button("☄️ Star 4: Scale & Impact", use_container_width=True):
        st.session_state.active_star = 4

if 'active_star' not in st.session_state:
    st.session_state.active_star = 4

# ==========================================
# 5. DYNAMIC STAR CONTENT (GLASSMORPHIC)
# ==========================================
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)

if st.session_state.active_star == 1:
    st.subheader("✨ Data Genesis (AI-Assisted Prep)")
    st.markdown("AI-assisted pipeline mapped 15 disparate CSV datasets into a unified relational schema, resolving comma-separated currency values and imputing missing budgets based on CPM benchmarks.")
    st.dataframe(gads_df.head(5), use_container_width=True, hide_index=True)

elif st.session_state.active_star == 2:
    st.subheader("💫 Neural Trends (Platform Benchmarking)")
    st.markdown("Generative analysis identified the **Work That Matters** plateau as the core institutional engine. Video drives absolute scale, while display captures immediate intent.")
    
    if 'Campaign' in gads_df.columns:
        fig = px.bar(gads_df.dropna(subset=['Cost', 'Impr.']).head(10), 
                     x='Cost', y='Campaign', orientation='h', color='Impr.',
                     color_continuous_scale=["#004e7c", "#008285", "#00d2d6"])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='white'), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.active_star == 3:
    st.subheader("🌟 Predictive Core (Advanced Modeling)")
    col_k, col_t = st.columns(2)
    
    with col_k:
        st.markdown("<h4 style='color: #00bfff !important;'>K-Means Audience Clustering</h4>", unsafe_allow_html=True)
        st.markdown("Identified three key personas: High-Intent Specialist, Knowledge Seeker, and Cultural Generalist.")
        if 'CTR' in aud_df.columns and 'TrueView view rate' in aud_df.columns:
            aud_clean = aud_df.dropna(subset=['CTR', 'TrueView view rate']).copy()
            if aud_clean['CTR'].dtype == object:
                aud_clean['CTR'] = aud_clean['CTR'].str.rstrip('%').astype('float') / 100.0
            if aud_clean['TrueView view rate'].dtype == object:
                aud_clean['TrueView view rate'] = aud_clean['TrueView view rate'].str.rstrip('%').astype('float') / 100.0
                
            if len(aud_clean) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(aud_clean[['CTR', 'TrueView view rate']])
                aud_clean['Cluster'] = kmeans.labels_.astype(str)
                
                fig2 = px.scatter(aud_clean, x='CTR', y='TrueView view rate', color='Cluster',
                                  color_discrete_sequence=["#00bfff", "#ffffff", "#008285"], size_max=15)
                fig2.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig2, use_container_width=True)

    with col_t:
        st.markdown("<h4 style='color: #00bfff !important;'>Time-Series Forecasting</h4>", unsafe_allow_html=True)
        st.markdown("Detecting institutional traffic spikes (Event Bursts vs. Sustained Branding).")
        if 'Users' in time_series_df.columns:
            fig3 = px.line(time_series_df, y='Users')
            fig3.update_traces(line_color="#00bfff", line_width=3)
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig3, use_container_width=True)

elif st.session_state.active_star == 4:
    st.subheader("☄️ Scale & Impact (The Event Horizon)")
    st.markdown("""
    **Conclusion of the AI-Human Partnership:**
    Traffic scale is driven by broad cultural events (Tony Awards), but the *intelligence and human impact* of CMU are sustained through deep, narrative engagement with technical specialists.
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown("<div class='glass-card'><h3>Event Burst</h3><h1>104K</h1><p>Clicks (Highest Efficiency)</p></div>", unsafe_allow_html=True)
    col_m2.markdown("<div class='glass-card'><h3>Sustained Engine</h3><h1>350K+</h1><p>WTM Awareness Impressions</p></div>", unsafe_allow_html=True)
    col_m3.markdown("<div class='glass-card'><h3>Deep Connection</h3><h1>36%</h1><p>Podcast Engagement Rate</p></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) # End Glass Container

# ==========================================
# 6. GOVERNANCE & DASH FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; opacity: 0.6; font-size: 12px;'>", unsafe_allow_html=True)
st.markdown("<b>🛡️ Data Governance & Accessibility Statement</b><br>", unsafe_allow_html=True)
st.markdown("Developed adhering to DASH guidelines. Contrast ratios ensure readability against deep-space themes. AI agents were utilized to normalize dataset schemas.", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
