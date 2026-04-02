import streamlit as st
import pandas as pd
import time

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Data Journey | CMU", page_icon="🕳️", layout="wide")

if 'current_location' not in st.session_state:
    st.session_state.current_location = 'galaxy'

def warp_to(location):
    st.session_state.current_location = location

# CMU Official Colors
CMU_RED = "#C41230"
CMU_IRON = "#6D6E71"
CMU_BLACK = "#000000"

# ==========================================
# 2. DYNAMIC CSS (BLACK HOLE & MILKY WAY)
# ==========================================
bg_positions = {
    'galaxy': "background-size: 100% 100%; background-position: center;",
    'star_1': "background-size: 250% 250%; background-position: 10% 20%;",
    'star_2': "background-size: 250% 250%; background-position: 80% 30%;",
    'star_3': "background-size: 250% 250%; background-position: 20% 80%;",
    'star_4': "background-size: 250% 250%; background-position: 90% 90%;"
}

current_bg_css = bg_positions[st.session_state.current_location]

st.markdown(f"""
    <style>
    /* The Black Hole + Milky Way Background */
    .stApp {{
        background-color: {CMU_BLACK};
        background-image: 
            /* 1. The Black Hole Center (Absolute Black) */
            radial-gradient(circle at 50% 50%, #000000 0%, #000000 8%, transparent 10%),
            /* 2. The Accretion Disk (CMU Red Glow) */
            radial-gradient(circle at 50% 50%, rgba(196, 18, 48, 0.8) 10%, rgba(109, 110, 113, 0.4) 25%, transparent 50%),
            /* 3. The Milky Way Starfield */
            url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=3000&q=80");
        background-attachment: fixed;
        {current_bg_css}
        transition: background-size 1.5s ease-in-out, background-position 1.5s ease-in-out;
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
    }}
    
    header {{visibility: hidden;}}
    
    /* Perfect Circular Twinkling Stars (No Squares) */
    .distant-star {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    
    .distant-star > div > button {{
        width: 12px !important;
        height: 12px !important;
        min-width: 12px !important;
        min-height: 12px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        background: #ffffff !important;
        box-shadow: 0 0 8px #fff, 0 0 15px {CMU_RED} !important;
        border: none !important;
        color: transparent !important;
        transition: all 0.3s ease !important;
        animation: twinkle 2s infinite alternate;
    }}
    
    /* Remove standard Streamlit button hover artifacts */
    .distant-star > div > button:focus, .distant-star > div > button:active {{
        outline: none !important;
        box-shadow: 0 0 20px #fff, 0 0 40px {CMU_RED} !important;
    }}
    
    .distant-star > div > button:hover {{
        transform: scale(2) !important;
        box-shadow: 0 0 20px #fff, 0 0 40px {CMU_RED} !important;
        cursor: pointer;
        background: {CMU_RED} !important;
    }}
    
    @keyframes twinkle {{
        0% {{ opacity: 0.5; box-shadow: 0 0 4px #fff, 0 0 8px rgba(196, 18, 48, 0.5); }}
        100% {{ opacity: 1; box-shadow: 0 0 12px #fff, 0 0 20px {CMU_RED}; }}
    }}

    /* Tiny Labels beneath the stars */
    .tiny-label {{
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.65rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 8px;
        text-shadow: 0 0 5px #000;
        text-align: center;
    }}

    /* The Data Story Panel (CMU Branded Glassmorphism) */
    @keyframes fadeUp {{
        0% {{ opacity: 0; transform: translateY(40px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .story-panel {{
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 4px solid {CMU_RED};
        border-right: 1px solid rgba(109, 110, 113, 0.3);
        border-top: 1px solid rgba(109, 110, 113, 0.3);
        border-bottom: 1px solid rgba(109, 110, 113, 0.3);
        border-radius: 8px;
        padding: 40px;
        margin-top: 50px;
        animation: fadeUp 1s ease-out forwards;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }}
    
    /* Highlight text in CMU Red */
    .cmu-highlight {{
        color: {CMU_RED};
        font-weight: bold;
    }}
    
    /* Warp Button (CMU Branded) */
    .warp-btn > button {{
        background: transparent !important;
        border: 1px solid {CMU_RED} !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        letter-spacing: 1px;
        padding: 10px 30px !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
    }}
    .warp-btn > button:hover {{
        background: rgba(196, 18, 48, 0.2) !important;
        box-shadow: 0 0 15px rgba(196, 18, 48, 0.6) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. VIEW ROUTING
# ==========================================

if st.session_state.current_location == 'galaxy':
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; font-weight: 200; margin-top: 5vh; letter-spacing: 2px;'>The Event Horizon</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1rem; color: {CMU_IRON};'>A vast expanse of raw data pulled by the gravity of our mission. Click a star to begin.</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    with c2:
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s1"): warp_to('star_1')
        st.markdown("<div class='tiny-label'>Genesis</div></div>", unsafe_allow_html=True)
        
    with c4:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s2"): warp_to('star_2')
        st.markdown("<div class='tiny-label'>Patterns</div></div>", unsafe_allow_html=True)

    with c1:
        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s3"): warp_to('star_3')
        st.markdown("<div class='tiny-label'>Gravity</div></div>", unsafe_allow_html=True)

    with c6:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button(" ", key="s4"): warp_to('star_4')
        st.markdown("<div class='tiny-label'>Horizon</div></div>", unsafe_allow_html=True)

elif st.session_state.current_location == 'star_1':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR I</h5>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0; font-weight: 300;'>Genesis: The Raw Export</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    This is our foundational dataset containing the financial and operational lifeblood of our campaigns: Budgets, Impressions, Costs, and CTRs.
    
    **The Messy Reality:**
    When we first arrived, it was chaotic. Costs and Impressions were exported as string text with commas (e.g., <span class='cmu-highlight'>"83,130.24"</span>), and empty points were represented as `--`.
    
    **How We Purified It:**
    We built an automated Python parser using regex to strip out commas, converting anomalies to mathematical floats. Our analytical engines can now read the data perfectly.
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Patterns ➔", use_container_width=True): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_location == 'star_2':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR II</h5>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0; font-weight: 300;'>Patterns: Time Series</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    Here we observe the daily ebb and flow of human traffic hitting our landing pages, directly tied to specific marketing campaigns.
    
    **The Messy Reality:**
    The data was exported in a "Wide Format". Instead of a simple timeline, we had a single row for a campaign, followed by 54+ columns labeled <span class='cmu-highlight'>Total users_Day0</span>, stretching infinitely to the right.
    
    **How We Purified It:**
    We utilized a data transformation technique called "Melting". We unpivoted the wide table, allowing us to map the precise day an event burst occurred.
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back", use_container_width=True): warp_to('star_1')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Gravity ➔", use_container_width=True): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_location == 'star_3':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR III</h5>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0; font-weight: 300;'>Gravity: Audience Intent</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    Deep within this sector, we find the psychological profiles of our users. It categorizes who is clicking—from "Arts Aficionados" to "Deep Learning Specialists."
    
    **The Messy Reality:**
    This sector was filled with noise. The largest segment was labeled <span class='cmu-highlight'>(not set)</span>. Engagement metrics were hardcoded as strings with percent signs (e.g., <span class='cmu-highlight'>5.85%</span>).
    
    **How We Purified It:**
    We filtered out the dark matter and stripped the `%` signs to convert the text into mathematical floats. This allowed us to feed the clean data into a K-Means clustering algorithm.
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back", use_container_width=True): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Next: Horizon ➔", use_container_width=True): warp_to('star_4')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_location == 'star_4':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='color: {CMU_RED}; letter-spacing: 3px; margin-bottom: 0;'>STAR IV</h5>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0; font-weight: 300;'>The Event Horizon</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    This is the culmination of our journey. Here, we bring together the cleansed financial data, the structured time-series, and the behavioral audience clusters to view the total impact of CMU's global outreach.
    
    **The Final Synthesis:**
    Because we spent the time meticulously cleaning commas, handling nulls, and melting date columns, our dashboard can now render this reality in real-time. The data is no longer raw chaos; it is a clear, navigable constellation pulled perfectly into focus by the gravity of Carnegie Mellon's mission.
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅ Back", use_container_width=True): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("🌌 Zoom Out", use_container_width=True): warp_to('galaxy')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
