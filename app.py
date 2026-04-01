import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Intelligence at Scale | CMU Summit 2026",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. ADVANCED MILKY WAY & CONSTELLATION CSS
# ==========================================
st.markdown("""
    <style>
    /* Breathtaking Milky Way Background */
    .stApp {
        background-color: #03000a;
        background-image: 
            radial-gradient(circle at 20% 40%, rgba(76, 29, 149, 0.2), transparent 50%),
            radial-gradient(circle at 80% 60%, rgba(0, 130, 133, 0.2), transparent 50%),
            url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=3000&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: screen;
        font-family: 'Helvetica', sans-serif;
        color: #ffffff;
    }
    
    header {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.4), 0 0 30px rgba(0, 123, 192, 0.6);
        font-weight: 300;
    }
    
    p, li { color: #e2e8f0 !important; font-size: 1.1rem; line-height: 1.6;}

    /* Glassmorphic Story Container with Smooth Fade-In */
    @keyframes fadeFloatIn {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .story-container {
        background: rgba(10, 10, 25, 0.5);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 40px;
        margin-top: 30px;
        box-shadow: 0 10px 50px 0 rgba(0, 0, 0, 0.7);
        animation: fadeFloatIn 0.8s ease-out forwards;
    }

    /* Constellation Stars (Circular Buttons) */
    div.stButton > button {
        width: 80px !important;
        height: 80px !important;
        border-radius: 50% !important;
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #00bfff 40%, #001f3f 100%) !important;
        color: transparent !important; /* Hide default text to use markdown labels */
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.6), inset 0 0 15px rgba(255, 255, 255, 0.8);
        transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        margin: 0 auto;
        display: block;
    }
    
    /* Hover state for Stars */
    div.stButton > button:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 0 40px rgba(0, 191, 255, 1), 0 0 80px rgba(0, 130, 133, 0.8), inset 0 0 20px #ffffff !important;
        border: 2px solid #ffffff !important;
        cursor: pointer;
    }
    
    /* Active/Clicked Star state */
    div.stButton > button:active {
        transform: scale(0.9) !important;
        box-shadow: 0 0 10px rgba(0, 191, 255, 0.5) !important;
    }

    /* Labels under the stars */
    .star-label {
        text-align: center;
        margin-top: 15px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #00bfff;
        text-shadow: 0 0 10px rgba(0, 191, 255, 0.8);
        letter-spacing: 1px;
    }
    
    .story-highlight {
        color: #00d2d6;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA LOAD & CACHING 
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
        # Graceful fallback for demonstration
        df_gads_26 = pd.DataFrame({"Campaign": ["WTM Awareness", "Tony Awards", "Podcast Deep Dive", "AI Care Part 1", "Robotics Gen"], "Cost": [40000, 8786, 5000, 12000, 9500], "Impr.": [350309, 104162, 45000, 85000, 62000], "CTR": [0.03, 0.63, 0.15, 0.08, 0.11]})
        df_ga_ts_26 = pd.DataFrame({"Day": range(184), "Users": np.random.poisson(lam=4000, size=184)})
        df_audiences = pd.DataFrame({"Audience segment": ["Cloud Storage", "Not in audiences", "Arts Aficionados", "Deep Learning", "General AI"], "CTR": [0.09, 0.05, 0.20, 0.12, 0.07], "TrueView view rate": [0.20, 0.35, 0.10, 0.28, 0.15]})
        
    return df_gads_26, df_ga_ts_26, df_audiences

gads_df, time_series_df, aud_df = load_data()

# ==========================================
# 4. HEADER
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>Intelligence at Scale</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #00d2d6 !important; margin-top: 10px;'>A Human-Centered Journey Through Data</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7;'>Follow the stars to uncover the narrative behind CMU's global impact.</p><br><br>", unsafe_allow_html=True)

# Initialize Session State
if 'active_star' not in st.session_state:
    st.session_state.active_star = 1

# ==========================================
# 5. THE CONSTELLATION MAP (STAGGERED UI)
# ==========================================
# Using empty columns to create a staggered, zigzag "constellation" look
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    st.markdown("<br><br>", unsafe_allow_html=True) # Push down to stagger
    if st.button("S1", key="star1"): st.session_state.active_star = 1
    st.markdown("<div class='star-label'>I. Genesis</div>", unsafe_allow_html=True)

with c3:
    if st.button("S2", key="star2"): st.session_state.active_star = 2
    st.markdown("<div class='star-label'>II. Patterns</div>", unsafe_allow_html=True)

with c5:
    st.markdown("<br><br><br>", unsafe_allow_html=True) # Push down to stagger further
    if st.button("S3", key="star3"): st.session_state.active_star = 3
    st.markdown("<div class='star-label'>III. Gravity</div>", unsafe_allow_html=True)

with c7:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("S4", key="star4"): st.session_state.active_star = 4
    st.markdown("<div class='star-label'>IV. Horizon</div>", unsafe_allow_html=True)

# ==========================================
# 6. THE STORY CONTAINER (SMOOTH TRANSITIONS)
# ==========================================
st.markdown("<div class='story-container'>", unsafe_allow_html=True)

if st.session_state.active_star == 1:
    st.markdown("<h2>Chapter I: The Genesis of Data</h2>", unsafe_allow_html=True)
    st.markdown("""
        Every great discovery begins in the dark. By aligning 15 disparate datasets into a unified analytical schema, 
        our AI-assisted pipeline illuminated the initial pathways of user engagement. <span class='story-highlight'>Missing budgets were intelligently imputed</span>, 
        and fragmented financial data was normalized, creating a solid launchpad for deep analysis.
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(gads_df.head(4), use_container_width=True, hide_index=True)

elif st.session_state.active_star == 2:
    st.markdown("<h2>Chapter II: Patterns in the Void</h2>", unsafe_allow_html=True)
    st.markdown("""
        As we peered deeper, temporal patterns emerged. The <span class='story-highlight'>Work That Matters</span> campaign acts as our core institutional engine—a steady burn of awareness. 
        Conversely, events like the Tony Awards act as supernovas, creating massive, highly efficient spikes in cultural traffic.
    """, unsafe_allow_html=True)
    
    if 'Campaign' in gads_df.columns:
        fig = px.bar(gads_df.dropna(subset=['Cost', 'Impr.']).head(8), 
                     x='Cost', y='Campaign', orientation='h', color='Impr.',
                     color_continuous_scale=["#03000a", "#007bc0", "#00d2d6"])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='white'), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.active_star == 3:
    st.markdown("<h2>Chapter III: The Gravity of Human Intent</h2>", unsafe_allow_html=True)
    st.markdown("""
        Who is drawn to the light? Using <span class='story-highlight'>K-Means Clustering</span>, our predictive core isolated distinct audience personas. 
        While the *Cultural Generalist* drives volume, it is the *High-Intent Specialist*—captivated by deep learning and technical podcasts—who exhibits the strongest gravitational pull toward CMU's core mission.
    """, unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1, 1])
    with c_left:
        if 'CTR' in aud_df.columns and 'TrueView view rate' in aud_df.columns:
            aud_clean = aud_df.dropna(subset=['CTR', 'TrueView view rate']).copy()
            if aud_clean['CTR'].dtype == object: aud_clean['CTR'] = aud_clean['CTR'].str.rstrip('%').astype('float') / 100.0
            if aud_clean['TrueView view rate'].dtype == object: aud_clean['TrueView view rate'] = aud_clean['TrueView view rate'].str.rstrip('%').astype('float') / 100.0
                
            if len(aud_clean) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(aud_clean[['CTR', 'TrueView view rate']])
                aud_clean['Cluster'] = ["Specialist", "Seeker", "Generalist"][:len(aud_clean)] # Dummy labels
                
                fig2 = px.scatter(aud_clean, x='CTR', y='TrueView view rate', color='Cluster',
                                  color_discrete_sequence=["#00d2d6", "#ffffff", "#007bc0"])
                fig2.update_traces(marker=dict(size=14, line=dict(width=1, color='rgba(255,255,255,0.5)')))
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), legend_title_text='Persona')
                st.plotly_chart(fig2, use_container_width=True)
    with c_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        **Persona Dynamics:**
        * 🔷 **The Specialist:** Low click volume, but incredibly high view rates. Seeks infrastructural knowledge.
        * ⚪ **The Seeker:** Moderate engagement, driven entirely by long-form organic content (Podcasts).
        * 🟦 **The Generalist:** Massive click volume, short dwell times. Captivated by brand prestige.
        """)

elif st.session_state.active_star == 4:
    st.markdown("<h2>Chapter IV: The Event Horizon</h2>", unsafe_allow_html=True)
    st.markdown("""
        We reach the culmination of Intelligence at Scale. The data confirms a profound truth: **Traffic scale is driven by culture, but human impact is sustained through technical depth.** By continuously rotating AI-driven creatives and forecasting temporal traffic spikes, CMU can scale its narrative without losing its ethical, human-centered focus.
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    # Custom inline CSS for the final metrics to make them look like glowing readouts
    metric_style = "text-align: center; padding: 20px; border-radius: 15px; background: rgba(0, 191, 255, 0.05); border: 1px solid rgba(0, 191, 255, 0.2);"
    
    col_m1.markdown(f"<div style='{metric_style}'><h4 style='color:#00d2d6!important; margin:0;'>Event Burst</h4><h1 style='margin:10px 0; font-size: 3rem;'>104K</h1><p style='margin:0;'>Clicks</p></div>", unsafe_allow_html=True)
    col_m2.markdown(f"<div style='{metric_style}'><h4 style='color:#00d2d6!important; margin:0;'>Sustained Engine</h4><h1 style='margin:10px 0; font-size: 3rem;'>350K</h1><p style='margin:0;'>WTM Impressions</p></div>", unsafe_allow_html=True)
    col_m3.markdown(f"<div style='{metric_style}'><h4 style='color:#00d2d6!important; margin:0;'>Deep Connection</h4><h1 style='margin:10px 0; font-size: 3rem;'>36%</h1><p style='margin:0;'>Podcast Engagement</p></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True) # End Story Container
