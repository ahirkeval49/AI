import streamlit as st
import pandas as pd
import time

# ==========================================
# 1. PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Data Journey | CMU", page_icon="🔭", layout="wide")

# Initialize our journey state
if 'current_location' not in st.session_state:
    st.session_state.current_location = 'galaxy' # Options: 'galaxy', 'star_1', 'star_2', 'star_3', 'star_4'

# Function to handle navigation
def warp_to(location):
    st.session_state.current_location = location

# ==========================================
# 2. DYNAMIC CSS (THE "3D" CAMERA ENGINE)
# ==========================================
# We change the background size and position based on where the user is in the story
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
    /* The Dynamic Galaxy Background */
    .stApp {{
        background-color: #010005;
        background-image: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.5)), url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?ixlib=rb-4.0.3&auto=format&fit=crop&w=3000&q=80");
        background-attachment: fixed;
        {current_bg_css}
        transition: background-size 1.5s ease-in-out, background-position 1.5s ease-in-out;
        color: #ffffff;
    }}
    
    header {{visibility: hidden;}}
    
    /* Distant Twinkling Stars (No longer big blue circles) */
    .distant-star > button {{
        width: 15px !important;
        height: 15px !important;
        border-radius: 50% !important;
        background: #ffffff !important;
        box-shadow: 0 0 10px #fff, 0 0 20px #00bfff, 0 0 30px #00bfff !important;
        border: none !important;
        color: transparent !important;
        transition: all 0.3s ease !important;
        animation: twinkle 3s infinite alternate;
    }}
    .distant-star > button:hover {{
        transform: scale(2) !important;
        box-shadow: 0 0 20px #fff, 0 0 40px #00d2d6 !important;
        cursor: pointer;
    }}
    
    @keyframes twinkle {{
        0% {{ opacity: 0.6; box-shadow: 0 0 5px #fff, 0 0 10px #00bfff; }}
        100% {{ opacity: 1; box-shadow: 0 0 15px #fff, 0 0 25px #00bfff; }}
    }}

    /* The Data Story Panel (Glassmorphic & Immersive) */
    @keyframes fadeUp {{
        0% {{ opacity: 0; transform: translateY(40px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .story-panel {{
        background: rgba(5, 5, 12, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-left: 3px solid #00bfff;
        border-radius: 10px;
        padding: 40px;
        margin-top: 50px;
        animation: fadeUp 1s ease-out forwards;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }}
    
    /* Warp Button */
    .warp-btn > button {{
        background: transparent !important;
        border: 1px solid #00d2d6 !important;
        color: #00d2d6 !important;
        font-size: 1.2rem !important;
        padding: 10px 30px !important;
        border-radius: 30px !important;
        transition: all 0.3s ease !important;
    }}
    .warp-btn > button:hover {{
        background: rgba(0, 210, 214, 0.2) !important;
        box-shadow: 0 0 15px rgba(0, 210, 214, 0.5) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. VIEW ROUTING
# ==========================================

# --- VIEW 1: THE OBSERVATORY (GALAXY MAP) ---
if st.session_state.current_location == 'galaxy':
    st.markdown("<h1 style='text-align: center; font-size: 4rem; font-weight: 200; margin-top: 5vh;'>The CMU Galaxy</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; opacity: 0.8;'>A vast expanse of raw data. Do you see the bright stars? Click the first to begin the journey.</p>", unsafe_allow_html=True)
    
    # Spacing to map out the constellation pseudo-randomly
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    with c2:
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button("1", key="s1"): warp_to('star_1')
        st.markdown("</div><p style='color:#00bfff; font-size:0.9rem; margin-top:10px;'>I. Genesis</p>", unsafe_allow_html=True)
        
    with c4:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button("2", key="s2"): warp_to('star_2')
        st.markdown("</div><p style='color:#00bfff; font-size:0.9rem; margin-top:10px;'>II. Patterns</p>", unsafe_allow_html=True)

    with c1:
        st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button("3", key="s3"): warp_to('star_3')
        st.markdown("</div><p style='color:#00bfff; font-size:0.9rem; margin-top:10px;'>III. Gravity</p>", unsafe_allow_html=True)

    with c6:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='distant-star'>", unsafe_allow_html=True)
        if st.button("4", key="s4"): warp_to('star_4')
        st.markdown("</div><p style='color:#00bfff; font-size:0.9rem; margin-top:10px;'>IV. Horizon</p>", unsafe_allow_html=True)


# --- VIEW 2: STAR 1 (GENESIS) ---
elif st.session_state.current_location == 'star_1':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00bfff; letter-spacing: 2px;'>STAR I</h4>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0;'>Genesis: The Raw Google Ads Export</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    This is our foundational dataset (`GAds_FY26_Totals`). It contains the financial and operational lifeblood of our campaigns: Budgets, Impressions, Costs, and Click-Through Rates across all Search, Display, and Video efforts.
    
    **The Messy Reality (Issues):**
    When we first arrived here, it was chaotic. 
    * Google Ads exports include meaningless summary rows at the top and bottom (`Total: Campaigns`, `Total: Account`).
    * Costs and Impressions were exported as string text with commas (e.g., `"83,130.24"` instead of `83130.24`).
    * Empty data points were represented as `--` instead of proper null values, breaking standard mathematical operations.
    
    **How We Purified It:**
    We built an automated Python parser. It skips the preamble rows, uses Regular Expressions (Regex) to strip out commas and currency symbols, and replaces all `--` anomalies with `NaN` (Not a Number). Finally, it casts these columns into strict float values, allowing our analytical engines to read them perfectly.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Warp to Star II: Patterns 🚀", use_container_width=True): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- VIEW 3: STAR 2 (PATTERNS) ---
elif st.session_state.current_location == 'star_2':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00bfff; letter-spacing: 2px;'>STAR II</h4>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0;'>Patterns: Google Analytics Time Series</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    Here we observe `GA_FY26_TimeSeries`. This file tracks the daily ebb and flow of human traffic hitting our landing pages, directly tied to specific marketing campaigns.
    
    **The Messy Reality (Issues):**
    The data was exported in a "Wide Format". Instead of a simple timeline, we had a single row for a campaign, followed by 54+ columns labeled `Total users_Day0`, `Total users_Day1`, `Total users_Day2`, stretching infinitely to the right. You cannot build a time-series graph with data structured like this.
    
    **How We Purified It:**
    We utilized a data transformation technique called "Melting". We unpivoted the wide table, transforming those 54 day-columns into two clean variables: `Day_Index` and `User_Count`. This allowed us to map the precise day an event burst (like the Tony Awards) occurred and separate it from our sustained, baseline traffic patterns.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Genesis", use_container_width=True): warp_to('star_1')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Warp to Star III: Gravity 🚀", use_container_width=True): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- VIEW 4: STAR 3 (GRAVITY) ---
elif st.session_state.current_location == 'star_3':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00bfff; letter-spacing: 2px;'>STAR III</h4>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0;'>Gravity: Audience Intent</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    Deep within `GAds_AudiencePerformance`, we find the psychological profiles of our users. It categorizes who is clicking—from "Arts Aficionados" to "Deep Learning Specialists."
    
    **The Messy Reality (Issues):**
    This sector was filled with noise. The largest audience segment by volume was simply labeled `People not in audiences` or `(not set)`. Furthermore, engagement metrics like Click-Through Rate (CTR) and TrueView rates were hardcoded as strings with percent signs (e.g., `5.85%`), preventing any algorithmic clustering.
    
    **How We Purified It:**
    First, we filtered out the dark matter—removing all `(not set)` and `not in audience` rows to focus only on identifiable human personas. Second, we stripped the `%` signs and divided by 100 to convert the text into mathematical floats (e.g., `0.0585`). This allowed us to feed the clean data into a K-Means clustering algorithm, revealing that technical specialists possess the strongest "gravitational pull" toward our content.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Patterns", use_container_width=True): warp_to('star_2')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("Warp to Star IV: Horizon 🚀", use_container_width=True): warp_to('star_4')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --- VIEW 5: STAR 4 (HORIZON) ---
elif st.session_state.current_location == 'star_4':
    st.markdown("<div class='story-panel'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00bfff; letter-spacing: 2px;'>STAR IV</h4>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: 0;'>The Event Horizon: Final Aggregation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    **What this data holds:**
    This is the culmination of our journey. Here, we bring together the cleansed financial data, the structured time-series, and the behavioral audience clusters to view the total impact of CMU's global outreach.
    
    **The Final Synthesis:**
    Because we spent the time meticulously cleaning commas, handling nulls, and melting date columns, our dashboard can now render this reality in real-time. 
    
    We can definitively prove that while broad cultural events create massive bursts of traffic, it is the sustained, highly-targeted technical content that drives the deepest human connection and engagement rates. The data is no longer raw chaos; it is a clear, navigable constellation.
    """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Gravity", use_container_width=True): warp_to('star_3')
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='warp-btn'>", unsafe_allow_html=True)
        if st.button("🌌 Return to Galaxy View", use_container_width=True): warp_to('galaxy')
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
