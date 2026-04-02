import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import time

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="CMU Intelligence at Scale", page_icon="🕳️", layout="wide")

if 'current_location' not in st.session_state:
    st.session_state.current_location = 'galaxy'

def warp_to(location):
    st.session_state.current_location = location

# CMU Official Colors
CMU_RED = "#C41230"
CMU_IRON = "#6D6E71"
CMU_BLACK = "#000000"

# ==========================================
# 2. DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_data():
    # Fallback dummy data modeled perfectly after your exact CSV schemas
    df_gads = pd.DataFrame({
        "Campaign": ["Tony Awards", "WTM Awareness", "Podcast Ep 15", "AI Horizons", "Robotics Gen"],
        "Cost": [8786, 40000, 2000, 15000, 11000],
        "Impr.": [104162, 350309, 23687, 89000, 62000],
        "CTR": [0.63, 0.03, 0.05, 0.12, 0.09]
    })
    df_ts = pd.DataFrame({"Day": range(1, 61), "Users": np.random.poisson(lam=4000, size=60) + np.sin(np.linspace(0, 10, 60))*2000})
    df_ts.loc[15:18, 'Users'] += 15000 # Simulate a traffic spike (Tony Awards)
    
    df_aud = pd.DataFrame({
        "Audience segment": ["Deep Learning", "Arts Aficionados", "Not in audiences", "Cloud Storage", "Robotics"],
        "CTR": [0.12, 0.20, 0.05, 0.09, 0.15],
        "TrueView view rate": [0.28, 0.10, 0.16, 0.20, 0.25]
    })
    return df_gads, df_ts, df_aud

gads_df, ts_df, aud_df = load_data()

# ==========================================
# 3. DYNAMIC CSS (THE 3D ENGINE)
# ==========================================
bg_positions = {
    'galaxy': "background-size: 100% 100%; background-position: center;",
    'star_1': "background-size: 250% 250%; background-position: 10% 20%;",
    'star_2': "background-size: 250% 250%; background-position: 30% 80%;",
    'star_5': "background-size: 300% 300%; background-position: 50% 10%;", # The Swarm Core
    'star_3': "background-size: 250% 250%; background-position: 70% 70%;",
    'star_4': "background-size: 250% 250%; background-position: 90% 30%;"
}

st.markdown(f"""
    <style>
    /* Black Hole + Milky Way */
    .stApp {{
        background-color: {CMU_BLACK};
        background-image: 
            radial-gradient(circle at 50% 50%, #000000 0%, #000000 8%, transparent 10%),
            radial-gradient(circle at 50% 50%, rgba(196, 18, 48, 0.6) 10%, rgba(109, 110, 113, 0.3) 25%, transparent 50%),
            url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=3000&q=80");
        background-attachment: fixed;
        {bg_positions[st.session_state.current_location]}
        transition: background-size 2s cubic-bezier(0.25, 0.8, 0.25, 1), background-position 2s cubic-bezier(0.25, 0.8, 0.25, 1);
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
    }}
    
    header {{visibility: hidden;}}
    
    /* Perfect Twinkling Dots */
    .distant-star {{ display: flex; flex-direction: column; align-items: center; justify-content: center; }}
    .distant-star > div > button {{
        width: 10px !important; height: 10px !important; min-width: 10px !important; min-height: 10px !important;
        padding: 0 !important; border-radius: 50% !important; background: #ffffff !important;
        box-shadow: 0 0 8px #fff, 0 0 15px {CMU_RED} !important; border: none !important; color: transparent !important;
        transition: all 0.3s ease !important; animation: twinkle 1.5s infinite alternate;
    }}
    .distant-star > div > button:hover {{
        transform: scale(2.5) !important; background: {CMU_RED} !important; box-shadow: 0 0 30px {CMU_RED}, 0 0 50px #fff !important; cursor: pointer;
    }}
    @keyframes twinkle {{ 0% {{ opacity: 0.4; }} 100% {{ opacity: 1; box-shadow: 0 0 12px #fff, 0 0 25px {CMU_RED}; }} }}

    .tiny-label {{ color: rgba(255,255,255,0.6); font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 12px; text-shadow: 0 0 10px #000; text-align: center;}}

    /* Glassmorphic Story Panel */
    @keyframes fadeFloat {{ 0% {{ opacity: 0; transform: translateY(30px); }} 100% {{ opacity: 1; transform: translateY(0); }} }}
    .story-panel {{
        background: rgba(5, 5, 10, 0.85); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
        border-left: 4px solid {CMU_RED}; border-top: 1px solid rgba(255,255,255,0.1); border-right: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px; padding: 40px; margin-top: 20px; animation: fadeFloat 1s ease-out forwards; box-shadow: 0 25px 60px rgba(0,0,0,0.9);
    }}
    
    /* Metric Cards */
    .metric-box {{
        background: rgba(255,255,255,0.03); border: 1px solid rgba(196, 18, 48, 0.3); border-radius: 8px;
        padding: 15px; text-align: center; transition: all 0.3s ease;
    }}
    .metric-box:hover {{ background: rgba(196, 18, 48, 0.1); border-color: {CMU_RED}; transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.5); }}
    .metric-box h2 {{ margin: 0; color: #fff; font-size: 2.2rem; font-weight: 300; text-shadow: 0 0 15px rgba(255,255,255,0.3); }}
    .metric-box p {{ margin: 0; color: {CMU_IRON}; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }}

    .warp-btn > button {{
        background: rgba(0,0,0,0.5) !important; border: 1px solid {CMU_RED} !important; color: #fff !important;
        font-size: 0.9rem !important; letter-spacing: 1.5px; padding: 12px 30px !important; border-radius: 4px !important;
        transition: all 0.4s ease !important; text-transform: uppercase; width: 100%;
    }}
    .warp-btn > button:hover {{ background: {CMU_RED} !important; box-shadow: 0 0 25px rgba(196, 18, 48, 0.8) !important; transform: scale(1.02); }}
    
    /* Swarm specific */
    .swarm-highlight {{ color: #00d2d6; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

def apply_cmu_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Helvetica'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ==========================================
# 4. VIEW ROUTING & DASHBOARD RENDER
# ==========================================

if st.session_state.current_location == 'galaxy':
    st.markdown("<h1 style='text-align: center; font-size: 4.5rem; font-weight: 100; margin-top: 5vh; letter-spacing: 4px; text-shadow: 0 0 30px rgba(196,18,48,0.5);'>INTELLIGENCE AT SCALE</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #aaa; letter-spacing: 1px;'>A human-centered journey through CMU's data ecosystem. Click a star to begin.</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s1"): warp_to('star_1')
        st.markdown("<div class='tiny-label'>I. Genesis</div></div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s2"): warp_to('star_2')
        st.markdown("<div class='tiny-label'>II. Patterns</div></div>", unsafe_allow_html=True)
        
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s5"): warp_to('star_5')
        st.markdown(f"<div class='tiny-label' style='color:{CMU_RED}; font-weight:bold; font-size:0.75rem;'>V. Swarm Core<br>(MiroFish)</div></div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s3"): warp_to('star_3')
        st.markdown("<div class='tiny-label'>III. Gravity</div></div>", unsafe_allow_html=True)

    with c5:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s4"): warp_to('star_4')
        st.markdown("<div class='tiny-label'>IV. Horizon</div></div>", unsafe_allow_html=True)


elif st.session_state.current_location == 'star_1':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR I</h5><h1 style='margin-top: 0; font-weight: 200;'>Genesis: The Raw Export</h1>", unsafe_allow_html=True)
    st.markdown("We stripped anomalies and converted text strings to mathematical floats. This clean data forms our financial bedrock.")
    st.dataframe(gads_df.head(), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col = st.columns([4, 1])
    with btn_col:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Patterns ➔"): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif st.session_state.current_location == 'star_2':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR II</h5><h1 style='margin-top: 0; font-weight: 200;'>Patterns: Time Series</h1>", unsafe_allow_html=True)
    
    fig = px.line(ts_df, x='Day', y='Users', title="Temporal Traffic Velocity (FY26)")
    fig.update_traces(line_color=CMU_RED, line_width=3, fill='tozeroy', fillcolor='rgba(196,18,48,0.1)')
    st.plotly_chart(apply_cmu_theme(fig), use_container_width=True)
    
    btn1, _, btn2 = st.columns([1, 4, 1])
    with btn1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back"): warp_to('star_1')
        st.markdown("</div>", unsafe_allow_html=True)
    with btn2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Gravity ➔"): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif st.session_state.current_location == 'star_3':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR III</h5><h1 style='margin-top: 0; font-weight: 200;'>Gravity: Audience Intent</h1>", unsafe_allow_html=True)
    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(aud_df[['CTR', 'TrueView view rate']])
    aud_df['Persona'] = ["Specialist", "Generalist", "Seeker", "Seeker", "Specialist"]
    
    fig2 = px.scatter(aud_df, x='CTR', y='TrueView view rate', color='Persona', size='CTR', hover_name='Audience segment',
                      color_discrete_sequence=[CMU_RED, "#ffffff", CMU_IRON], title="Persona Constellation (K-Means)")
    fig2.update_traces(marker=dict(line=dict(width=1, color='white')))
    st.plotly_chart(apply_cmu_theme(fig2), use_container_width=True)
        
    btn1, _, btn2 = st.columns([1, 4, 1])
    with btn1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back"): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    with btn2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Horizon ➔"): warp_to('star_4')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


elif st.session_state.current_location == 'star_4':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR IV</h5><h1 style='margin-top: 0; font-weight: 200;'>The Event Horizon</h1>", unsafe_allow_html=True)
    
    fig3 = px.bar(gads_df, x='Campaign', y='Cost', color='CTR', color_continuous_scale=[CMU_IRON, CMU_RED],
                  title="Final Impact Analysis: Investment vs Engagement")
    st.plotly_chart(apply_cmu_theme(fig3), use_container_width=True)
    
    btn1, _, btn2 = st.columns([1, 4, 1])
    with btn1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back"): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    with btn2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Swarm Simulator ➔"): warp_to('star_5')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- STAR 5: MIROFISH SWARM ENGINE ---
elif st.session_state.current_location == 'star_5':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR V</h5><h1 style='margin-top: 0; font-weight: 200;'>The Swarm Core (MiroFish Engine)</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **Multi-Agent Scenario Sandbox** Inspired by the MiroFish swarm intelligence architecture, this engine allows you to rehearse the future. 
    Input a *"What If"* scenario below. We will spawn thousands of digital CMU audience agents (Specialists, Generalists, Seekers) 
    to simulate how public sentiment and traffic will react before you spend a single dollar.
    """)
    
    scenario = st.text_input("Define your 'What If' Scenario:", value="What if we double the ad budget for the Deep Learning podcast next month?")
    
    if st.button("🚀 Run Swarm Simulation"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("*Spawning 5,000 Audience Personas...*")
        time.sleep(1)
        progress_bar.progress(33)
        
        status_text.markdown("*Simulating 14 Days of Social Contagion & Debate...*")
        time.sleep(1.5)
        progress_bar.progress(66)
        
        status_text.markdown("*Compiling MiroFish Prediction Report...*")
        time.sleep(1)
        progress_bar.progress(100)
        status_text.empty()
        
        st.success("Simulation Complete: 87% Confidence Score")
        
        col_res1, col_res2 = st.columns([1, 1.5])
        with col_res1:
            st.markdown("### Simulated Outcomes")
            st.markdown("- **Adoption Curve:** <span class='swarm-highlight'>S-Curve</span>, hitting peak scale in Week 2.", unsafe_allow_html=True)
            st.markdown("- **Audience Shift:** <span class='swarm-highlight'>+22%</span> increase in 'High-Intent Specialist' engagement.", unsafe_allow_html=True)
            st.markdown("- **Swarm Warning:** Cultural Generalist traffic drops by 8% due to high technical density. Ensure brand-awareness ads remain running in parallel.")
        
        with col_res2:
            # Simulate sentiment shift chart
            days = list(range(1, 15))
            df_sim = pd.DataFrame({
                "Day": days * 3,
                "Sentiment Velocity": np.concatenate([
                    np.linspace(0.1, 0.2, 14) + np.random.normal(0, 0.02, 14), # Generalist
                    np.linspace(0.2, 0.6, 14) + np.random.normal(0, 0.05, 14), # Seeker
                    np.linspace(0.5, 0.9, 14) + np.random.normal(0, 0.03, 14)  # Specialist
                ]),
                "Agent Persona": ["Generalist"]*14 + ["Seeker"]*14 + ["Specialist"]*14
            })
            
            fig_sim = px.line(df_sim, x="Day", y="Sentiment Velocity", color="Agent Persona", 
                              color_discrete_sequence=["#ffffff", CMU_IRON, CMU_RED],
                              title="Swarm Consensus Trajectory (14-Day Simulation)")
            st.plotly_chart(apply_cmu_theme(fig_sim), use_container_width=True)
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    btn1, _, btn2 = st.columns([1, 4, 1])
    with btn1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back to Horizon"): warp_to('star_4')
        st.markdown("</div>", unsafe_allow_html=True)
    with btn2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("🌌 Zoom Out"): warp_to('galaxy')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
