import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import streamlit.components.v1 as components

# Streamlit Extras for premium UI
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ROUTING SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CMU Campaign Intelligence", layout="wide", initial_sidebar_state="collapsed")

# Routing Logic: Read URL parameters to know which page to render
if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# ---------------------------------------------------------
# 2. CUSTOM UI: RADIAL MENU
# ---------------------------------------------------------
# Injects a floating CSS radial menu in the bottom right corner for global navigation
radial_menu_html = """
<style>
    .radial-nav { position: fixed; bottom: 40px; right: 40px; z-index: 999999; }
    .menu-button {
        width: 60px; height: 60px; background-color: #C41230; color: white;
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        font-size: 24px; cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: transform 0.3s ease;
    }
    .radial-nav:hover .menu-button { transform: rotate(45deg); }
    .menu-item {
        position: absolute; width: 45px; height: 45px; background-color: #333; color: white;
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        font-size: 11px; text-decoration: none; font-family: sans-serif; font-weight: bold;
        opacity: 0; transform: scale(0); transition: all 0.4s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3); border: 1px solid #555;
    }
    .radial-nav:hover .menu-item { opacity: 1; transform: scale(1); }
    .item-1 { bottom: 80px; right: 0px; }
    .item-2 { bottom: 65px; right: 65px; }
    .item-3 { bottom: 0px; right: 80px; }
    .menu-item:hover { background-color: #C41230; border-color: #FFF; color: #FFF;}
</style>
<div class="radial-nav">
    <div class="menu-button">✦</div>
    <a href="?page=home" target="_self" class="menu-item item-1" title="3D Home">3D</a>
    <a href="?page=data" target="_self" class="menu-item item-2" title="Data Hub">DATA</a>
    <a href="?page=roi" target="_self" class="menu-item item-3" title="ROI Matrix">ROI</a>
</div>
"""
st.markdown(radial_menu_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. DATA LOADING & CLEANING (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        index_df = pd.read_csv('data/UCM Campaign Index.csv')
        campaign_mgmt = pd.read_csv('data/2024-25_Campaign_Management_1769521985.csv')
        ga_time = pd.read_csv('data/GA_FY25_TimeSeries (1).csv')
        ga_utm = pd.read_csv('data/GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', skiprows=1)
        gads_perf = pd.read_csv('data/GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv')
        linkedin_perf = pd.read_csv('data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv')

        campaign_clean = campaign_mgmt.dropna(how='all')
        if 'Name' in campaign_clean.columns:
            campaign_clean = campaign_clean.dropna(subset=['Name'])

        melted_ga = pd.melt(ga_time, id_vars=['Session campaign', 'Segment'], var_name='Day', value_name='User_Count')
        melted_ga['Day_Number'] = melted_ga['Day'].str.extract(r'(\d+)').astype(float)
        melted_ga['User_Count'] = pd.to_numeric(melted_ga['User_Count'], errors='coerce').fillna(0)

        gads_perf.replace('--', np.nan, inplace=True)
        if 'Ad name' in gads_perf.columns:
            gads_perf = gads_perf[~gads_perf['Ad name'].astype(str).str.contains('Total', case=False, na=False)]
            
        for col in ['Clicks', 'Impr.', 'CTR', 'Cost']:
            if col in gads_perf.columns:
                gads_perf[col] = gads_perf[col].astype(str).str.replace(r'[,\%\$]', '', regex=True)
                gads_perf[col] = pd.to_numeric(gads_perf[col], errors='coerce')

        threshold = len(linkedin_perf) * 0.8
        linkedin_clean = linkedin_perf.dropna(thresh=len(linkedin_perf) - threshold, axis=1)
        
        return index_df, campaign_clean, melted_ga, ga_utm, gads_perf, linkedin_clean
        
    except FileNotFoundError as e:
        st.error(f"File missing: {e}. Please ensure all CSVs are in the 'data/' folder.")
        st.stop()

@st.cache_data
def create_master_view(index_df, ga_utm, gads_perf, linkedin_clean):
    if 'UTM campaign' in index_df.columns:
        index_df['utm_clean'] = index_df['UTM campaign'].astype(str).str.lower().str.strip()
    
    ga_utm_renamed = ga_utm.copy()
    if 'Session campaign' in ga_utm_renamed.columns:
        ga_utm_renamed['utm_clean'] = ga_utm_renamed['Session campaign'].astype(str).str.lower().str.strip()
        ga_agg = ga_utm_renamed.groupby('utm_clean').agg(
            Total_Website_Users=('Total users', 'sum'),
            Average_Engagement_Rate=('Engagement rate', 'mean')
        ).reset_index()
    else:
        ga_agg = pd.DataFrame(columns=['utm_clean', 'Total_Website_Users', 'Average_Engagement_Rate'])

    if 'Ad name' in gads_perf.columns and 'Cost' in gads_perf.columns:
        gads_perf['utm_clean'] = gads_perf['Ad name'].astype(str).str.lower().str.strip()
        gads_agg = gads_perf.groupby('utm_clean').agg(Total_GAds_Spend=('Cost', 'sum')).reset_index()
    else:
        gads_agg = pd.DataFrame(columns=['utm_clean', 'Total_GAds_Spend'])

    if 'Campaign Name' in linkedin_clean.columns and 'Total Spend' in linkedin_clean.columns:
        linkedin_clean['utm_clean'] = linkedin_clean['Campaign Name'].astype(str).str.lower().str.strip()
        li_agg = linkedin_clean.groupby('utm_clean').agg(Total_LinkedIn_Spend=('Total Spend', 'sum')).reset_index()
    else:
        li_agg = pd.DataFrame(columns=['utm_clean', 'Total_LinkedIn_Spend'])

    master_df = index_df.copy()
    if 'utm_clean' in master_df.columns:
        master_df = pd.merge(master_df, ga_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, gads_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, li_agg, on='utm_clean', how='left')
        master_df.fillna({'Total_GAds_Spend': 0, 'Total_LinkedIn_Spend': 0, 'Total_Website_Users': 0}, inplace=True)
        master_df['Total_Combined_Spend'] = master_df['Total_GAds_Spend'] + master_df['Total_LinkedIn_Spend']
        master_df['CPWU'] = master_df['Total_Combined_Spend'].div(master_df['Total_Website_Users'].replace(0, np.nan)).fillna(0)
    else:
        master_df['Total_Combined_Spend'], master_df['Total_Website_Users'], master_df['CPWU'] = 0, 0, 0
    
    return master_df

# Load datasets globally
index_df, campaign_clean, melted_ga, ga_utm, gads_perf, linkedin_clean = load_and_clean_data()
master_df = create_master_view(index_df, ga_utm, gads_perf, linkedin_clean)

# ---------------------------------------------------------
# 4. VIEW RENDERING (ROUTING)
# ---------------------------------------------------------

# ======================= HOME: 3D THREE.JS =======================
if current_page == "home":
    st.markdown("<h1 style='text-align: center; color: #C41230; font-family: sans-serif;'>CMU Data Nexus</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Drag to rotate camera. <b>Click the floating Agent Nodes</b> to explore the data.</p>", unsafe_allow_html=True)
    
    three_js_app = """
    <!DOCTYPE html>
    <html>
    <head>
        <style> body { margin: 0; overflow: hidden; background-color: #050505; border-radius: 10px; } </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
            renderer.setSize(window.innerWidth, 600);
            document.body.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

            // 1. Ambient Background Particles
            const particlesCount = 4000;
            const posArray = new Float32Array(particlesCount * 3);
            for(let i = 0; i < particlesCount * 3; i++) { posArray[i] = (Math.random() - 0.5) * 20; }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const material = new THREE.PointsMaterial({ size: 0.02, color: 0x555555 });
            const particlesMesh = new THREE.Points(geometry, material);
            scene.add(particlesMesh);

            // 2. Interactive Agent Nodes
            const agents = [];
            const sphereGeo = new THREE.SphereGeometry(0.5, 32, 32);
            
            // Agent 1: Data (Red)
            const mat1 = new THREE.MeshBasicMaterial({color: 0xC41230});
            const agent1 = new THREE.Mesh(sphereGeo, mat1);
            agent1.position.set(-3, 1, 2);
            agent1.userData = { url: "?page=data" }; 
            scene.add(agent1); agents.push(agent1);

            // Agent 2: ROI Matrix (Blue)
            const mat2 = new THREE.MeshBasicMaterial({color: 0x00A6D6});
            const agent2 = new THREE.Mesh(sphereGeo, mat2);
            agent2.position.set(3, -1, 1);
            agent2.userData = { url: "?page=roi" };
            scene.add(agent2); agents.push(agent2);

            camera.position.z = 6;

            // 3. Raycaster (Click Detection)
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            window.addEventListener('pointerdown', (event) => {
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / 600) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);
                if(intersects.length > 0) {
                    window.parent.location.search = intersects[0].object.userData.url;
                }
            });

            window.addEventListener('pointermove', (event) => {
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / 600) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);
                document.body.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
            });

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_app, height=600)

# ======================= DATA HUB =======================
elif current_page == "data":
    colored_header(label="Data Hub & Statistical Analysis", description="Explore cleaned telemetry and predictive modeling.", color_name="red-70")
    
    # KPIs with streamlit-extras styling
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Logged Campaigns", value=len(index_df))
    col2.metric(label="Total Cross-Platform Spend", value=f"${master_df['Total_Combined_Spend'].sum():,.2f}")
    col3.metric(label="Total Website Users Acquired", value=f"{master_df['Total_Website_Users'].sum():,.0f}")
    style_metric_cards(background_color="#1E1E1E", border_left_color="#C41230")
    
    st.divider()
    
    # Interactive Dataframe using streamlit-extras
    st.subheader("Master Data Explorer")
    filtered_df = dataframe_explorer(master_df, case=False)
    st.dataframe(filtered_df, use_container_width=True)

    st.divider()
    
    # Statistical Analysis
    st.subheader("Statistical Analysis")
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**1. Time-Series (OLS Regression)**")
        if not melted_ga.empty:
            top_campaigns = melted_ga.groupby('Session campaign')['User_Count'].sum().nlargest(5).index
            fig1 = px.scatter(
                melted_ga[melted_ga['Session campaign'].isin(top_campaigns)], 
                x="Day_Number", y="User_Count", color="Session campaign", trendline="ols", 
                template="plotly_dark"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
    with colB:
        st.markdown("**2. A/B Hypothesis Testing (T-Test)**")
        st.markdown("Testing if there is a statistically significant difference in engagement between Google vs. LinkedIn campaigns.")
        if 'Session source' in ga_utm.columns and 'Engagement rate' in ga_utm.columns:
            ga_utm['Engagement rate'] = pd.to_numeric(ga_utm['Engagement rate'], errors='coerce').fillna(0)
            google_eng = ga_utm[ga_utm['Session source'].str.contains('google', na=False, case=False)]['Engagement rate']
            linkedin_eng = ga_utm[ga_utm['Session source'].str.contains('linkedin', na=False, case=False)]['Engagement rate']
            
            if len(google_eng) > 0 and len(linkedin_eng) > 0:
                t_stat, p_val = stats.ttest_ind(google_eng, linkedin_eng, equal_var=False)
                st.info(f"**T-Statistic:** {t_stat:.4f} \n\n **P-Value:** {p_val:.4e}")
                if p_val < 0.05:
                    st.success("Result: The difference in engagement rates is statistically significant (p < 0.05).")
                else:
                    st.warning("Result: No statistically significant difference.")

# ======================= ROI MATRIX =======================
elif current_page == "roi":
    colored_header(label="Unified Cross-Platform ROI", description="Campaign efficiency evaluated across ad networks and on-site behavior.", color_name="light-blue-70")
    
    st.markdown("Ideal campaigns live in the **Top-Left quadrant** (High Users, Low Spend). Bubble size represents engagement rate.")
    
    valid_master = master_df[(master_df['Total_Website_Users'] > 0) & (master_df['Total_Combined_Spend'] > 0)]
    
    if not valid_master.empty:
        color_col = 'Category' if 'Category' in valid_master.columns else None
        fig2 = px.scatter(
            valid_master, x="Total_Combined_Spend", y="Total_Website_Users", 
            size="Average_Engagement_Rate", color=color_col,
            hover_name="Unique_Campaign_ID" if 'Unique_Campaign_ID' in valid_master.columns else None,
            labels={"Total_Combined_Spend": "Total Ad Spend ($)", "Total_Website_Users": "Website Users Acquired"},
            template="plotly_dark", height=600
        )
        
        avg_users, avg_spend = valid_master['Total_Website_Users'].mean(), valid_master['Total_Combined_Spend'].mean()
        fig2.add_hline(y=avg_users, line_dash="dot", annotation_text="Avg Users", annotation_position="top left")
        fig2.add_vline(x=avg_spend, line_dash="dot", annotation_text="Avg Spend", annotation_position="top right")
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough overlapping spend/user data to generate the ROI matrix. Check mapping keys.")
