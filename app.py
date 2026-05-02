import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from urllib.parse import urlparse, parse_qs

# Streamlit Extras for premium UI
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ROUTING SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus", layout="wide", initial_sidebar_state="collapsed")

# Routing Logic: Read URL parameters to know which page to render
if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# ---------------------------------------------------------
# 2. GLOBAL NAVIGATION: CLICKABLE DIAGRAM CARDS
# ---------------------------------------------------------
nav_cards_html = """
<style>
.nav-grid { display: flex; justify-content: center; gap: 20px; padding: 10px; flex-wrap: wrap; margin-bottom: 5px; position: relative; z-index: 100; }
.nav-card {
    background: #ffffff; border: 2px solid #E0E0E0; border-radius: 12px; padding: 15px 20px;
    width: 140px; text-align: center; color: #333 !important; text-decoration: none;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); font-family: sans-serif;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.nav-card:hover { 
    border-color: #C41230; transform: translateY(-8px) scale(1.05); 
    background: #FAFAFA; box-shadow: 0 10px 20px rgba(196,18,48,0.2); 
}
.nav-icon { font-size: 30px; margin-bottom: 8px; }
.nav-title { font-size: 12px; font-weight: bold; letter-spacing: 1px; color: #C41230; }
</style>
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card"><div class="nav-icon">🏠</div><div class="nav-title">HOME</div></a>
    <a href="?page=explorer" target="_self" class="nav-card"><div class="nav-icon">📊</div><div class="nav-title">EXPLORE</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card"><div class="nav-icon">🧹</div><div class="nav-title">CLEAN</div></a>
    <a href="?page=analysis" target="_self" class="nav-card"><div class="nav-icon">📈</div><div class="nav-title">STATS</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card"><div class="nav-icon">💡</div><div class="nav-title">DASHBOARD</div></a>
    <a href="?page=graph" target="_self" class="nav-card"><div class="nav-icon">🕸️</div><div class="nav-title">GRAPH</div></a>
</div>
"""

# ---------------------------------------------------------
# 3. GLOBAL DATA PROCESSING LOGIC
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM Campaign Index.csv"
]

@st.cache_data
def load_raw_file(filename):
    try:
        skip = 1 if 'UTM_Totals' in filename else 0
        df = pd.read_csv(f'data/{filename}', skiprows=skip)
        # Drop `.1` duplicate columns
        duplicate_cols = [c for c in df.columns if c.endswith('.1')]
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [f"Could not load {filename}: {str(e)}"]})

def extract_utm_campaign(url):
    try:
        parsed = urlparse(str(url))
        return parse_qs(parsed.query).get('utm_campaign', [np.nan])[0]
    except:
        return np.nan

@st.cache_data
def build_master_models():
    # Load core datasets
    index_df = load_raw_file("UCM Campaign Index.csv")
    ga_utm = pd.concat([load_raw_file("GA_FY25_UTM_Totals_Jul2024-Jun2025.csv"), load_raw_file("GA_FY26_UTM_Totals_Jul-Dec2025.csv")], ignore_index=True)
    ga_time = pd.concat([load_raw_file("GA_FY25_TimeSeries (1).csv"), load_raw_file("GA_FY26_TimeSeries.csv")], ignore_index=True)
    gads_perf = load_raw_file("GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv")
    linkedin_perf = load_raw_file("LinkedIn_Ad_Performance_Feb2024_Dec2025.csv")

    if 'Error' in index_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if 'UTM campaign' in index_df.columns and 'Landing Page (UTM)' in index_df.columns:
        index_df['UTM campaign'] = index_df['UTM campaign'].fillna(index_df['Landing Page (UTM)'].apply(extract_utm_campaign))
        index_df['utm_clean'] = index_df['UTM campaign'].astype(str).str.lower().str.strip()

    if 'Session campaign' in ga_time.columns:
        melted_ga = pd.melt(ga_time, id_vars=['Session campaign', 'Segment'], var_name='Day', value_name='User_Count')
        melted_ga['Day_Number'] = melted_ga['Day'].str.extract(r'(\d+)').astype(float)
        melted_ga['User_Count'] = pd.to_numeric(melted_ga['User_Count'], errors='coerce').fillna(0)
        melted_ga = melted_ga[melted_ga['User_Count'] > 0]
    else:
        melted_ga = pd.DataFrame()

    gads_perf.replace('--', np.nan, inplace=True)
    if 'Ad name' in gads_perf.columns:
        gads_perf = gads_perf[~gads_perf['Ad name'].astype(str).str.contains('Total', case=False, na=False)]
    for col in ['Clicks', 'Impr.', 'CTR', 'Cost']:
        if col in gads_perf.columns:
            gads_perf[col] = pd.to_numeric(gads_perf[col].astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce')

    if len(linkedin_perf) > 0 and 'Error' not in linkedin_perf.columns:
        linkedin_clean = linkedin_perf.dropna(thresh=len(linkedin_perf) * 0.2, axis=1)
    else:
        linkedin_clean = linkedin_perf

    ga_agg = pd.DataFrame(columns=['utm_clean', 'Total_Website_Users', 'Average_Engagement_Rate'])
    if 'Session campaign' in ga_utm.columns:
        ga_utm['utm_clean'] = ga_utm['Session campaign'].astype(str).str.lower().str.strip()
        ga_agg = ga_utm.groupby('utm_clean').agg(Total_Website_Users=('Total users', 'sum'), Average_Engagement_Rate=('Engagement rate', 'mean')).reset_index()

    gads_agg = pd.DataFrame(columns=['utm_clean', 'Total_GAds_Spend'])
    if 'Ad name' in gads_perf.columns and 'Cost' in gads_perf.columns:
        gads_perf['utm_clean'] = gads_perf['Ad name'].astype(str).str.lower().str.strip()
        gads_agg = gads_perf.groupby('utm_clean').agg(Total_GAds_Spend=('Cost', 'sum')).reset_index()

    li_agg = pd.DataFrame(columns=['utm_clean', 'Total_LinkedIn_Spend'])
    if 'Campaign Name' in linkedin_clean.columns and 'Total Spend' in linkedin_clean.columns:
        linkedin_clean['utm_clean'] = linkedin_clean['Campaign Name'].astype(str).str.lower().str.strip()
        li_agg = linkedin_clean.groupby('utm_clean').agg(Total_LinkedIn_Spend=('Total Spend', 'sum')).reset_index()

    master_df = index_df.copy()
    if 'utm_clean' in master_df.columns:
        master_df = pd.merge(master_df, ga_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, gads_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, li_agg, on='utm_clean', how='left')
        master_df.fillna({'Total_GAds_Spend': 0, 'Total_LinkedIn_Spend': 0, 'Total_Website_Users': 0}, inplace=True)
        master_df['Total_Combined_Spend'] = master_df['Total_GAds_Spend'] + master_df['Total_LinkedIn_Spend']
        master_df['CPWU'] = master_df['Total_Combined_Spend'].div(master_df['Total_Website_Users'].replace(0, np.nan)).fillna(0)
    
    return index_df, ga_utm, melted_ga, gads_perf, linkedin_clean, master_df

index_df, ga_utm, melted_ga, gads_perf, linkedin_clean, master_df = build_master_models()

# ---------------------------------------------------------
# 4. VIEW RENDERING (ROUTING)
# ---------------------------------------------------------

if current_page == "home":
    st.markdown("""
        <style>
            .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 0rem; padding-right: 0rem; max-width: 100%; }
            header { display: none !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 800; margin-top: -10px; font-size: 42px;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6D6E71;'>Explore the visual map. <b>Hover & click</b> a colored node.</p>", unsafe_allow_html=True)
    
    three_js_cmu_galaxy = """
    <!DOCTYPE html>
    <html>
    <head>
        <style> 
            body { margin: 0; overflow: hidden; background-color: #ffffff; font-family: sans-serif; } 
            canvas { display: block; width: 100vw; height: 100vh; }
            .node-tab {
                position: absolute; background: rgba(255, 255, 255, 0.95);
                border: 2px solid #C41230; padding: 6px 12px; border-radius: 8px;
                font-weight: bold; font-size: 13px; color: #222;
                pointer-events: none; transform: translate(-50%, -150%);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s, opacity 0.2s; opacity: 0.8;
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xffffff);

            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true; 
            controls.autoRotate = true; 
            controls.autoRotateSpeed = 0.5;

            // 1. CMU Brand Particle Galaxy
            const pCount = 15000;
            const pGeo = new THREE.BufferGeometry();
            const pos = new Float32Array(pCount * 3);
            const colors = new Float32Array(pCount * 3);

            const colorBlack = new THREE.Color(0x111111);
            const colorRed = new THREE.Color(0xC41230);
            const colorGray = new THREE.Color(0x6D6E71);
            const colorGold = new THREE.Color(0xE2C044);

            for(let i=0; i<pCount; i++) {
                const r = 25 * Math.cbrt(Math.random());
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                
                pos[i*3] = r * Math.sin(phi) * Math.cos(theta);
                pos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
                pos[i*3+2] = r * Math.cos(phi);

                const rand = Math.random();
                let c = colorBlack;
                if (rand > 0.70 && rand <= 0.85) c = colorRed;
                else if (rand > 0.85 && rand <= 0.95) c = colorGray;
                else if (rand > 0.95) c = colorGold;

                colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
            }
            pGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            pGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const pMat = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, transparent: true, opacity: 0.7 });
            const particleSystem = new THREE.Points(pGeo, pMat);
            scene.add(particleSystem);

            // 2. Tartan Mascot (Scotty Dog) Particles
            // Draw emoji to canvas to avoid external URL CORS/loading issues
            const scottyCanvas = document.createElement('canvas');
            scottyCanvas.width = 64; scottyCanvas.height = 64;
            const ctx = scottyCanvas.getContext('2d');
            ctx.font = '48px sans-serif';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText('🐕', 32, 32); 
            const scottyTexture = new THREE.CanvasTexture(scottyCanvas);

            const mascotCount = 200;
            const mGeo = new THREE.BufferGeometry();
            const mPos = new Float32Array(mascotCount * 3);
            for(let i=0; i<mascotCount; i++) {
                const r = 18 * Math.cbrt(Math.random());
                const theta = Math.random() * 2 * Math.PI;
                const phi = Math.acos(2 * Math.random() - 1);
                mPos[i*3] = r * Math.sin(phi) * Math.cos(theta);
                mPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
                mPos[i*3+2] = r * Math.cos(phi);
            }
            mGeo.setAttribute('position', new THREE.BufferAttribute(mPos, 3));
            const mMat = new THREE.PointsMaterial({ size: 0.8, map: scottyTexture, transparent: true, opacity: 0.8, alphaTest: 0.5 });
            const mascotSystem = new THREE.Points(mGeo, mMat);
            scene.add(mascotSystem);

            // 3. CMU Core Volumetric Letters
            const bGeo = new THREE.BoxGeometry(0.4, 0.4, 0.4);
            const bMat = new THREE.MeshStandardMaterial({ color: 0xC41230, roughness: 0.2, metalness: 0.3 });
            const core = new THREE.Group();
            const coords = [
                [0,0],[1,0],[2,0],[3,0],[0,-1],[0,-2],[0,-3],[0,-4],[1,-4],[2,-4],[3,-4],
                [5,0],[9,0],[5,-1],[6,-1],[8,-1],[9,-1],[5,-2],[7,-2],[9,-2],[5,-3],[9,-3],[5,-4],[9,-4],
                [11,0],[14,0],[11,-1],[14,-1],[11,-2],[14,-2],[11,-3],[14,-3],[11,-4],[12,-4],[13,-4],[14,-4]
            ];
            coords.forEach(p => {
                const m = new THREE.Mesh(bGeo, bMat);
                m.position.set(p[0]*0.5 - 3.5, p[1]*0.5 + 1, 0);
                core.add(m);
            });
            scene.add(core);

            // 4. Clickable Orbiting Nodes WITH HTML TABS
            const agents = [];
            const htmlTabs = [];
            const nodesConfig = [
                { name: "Explore", color: 0xE2C044, url: "?page=explorer" }, 
                { name: "Clean", color: 0xE87A5D, url: "?page=cleaner" },    
                { name: "Stats", color: 0x44BBA4, url: "?page=analysis" },   
                { name: "Dashboard", color: 0x00A6D6, url: "?page=dashboard" },
                { name: "Graph", color: 0x9B5DE5, url: "?page=graph" }       
            ];
            
            nodesConfig.forEach((config, i) => {
                const s = new THREE.Mesh(
                    new THREE.SphereGeometry(0.8, 32, 32), 
                    new THREE.MeshStandardMaterial({ color: config.color, roughness: 0.1, metalness: 0.4 })
                );
                const a = (i/5)*Math.PI*2;
                s.position.set(Math.cos(a)*7, Math.sin(a)*2, Math.sin(a)*7);
                s.userData = { url: config.url, angle: a }; 
                scene.add(s);
                agents.push(s);

                const tab = document.createElement('div');
                tab.className = 'node-tab';
                tab.innerText = config.name;
                tab.style.borderColor = '#' + config.color.toString(16).padStart(6, '0');
                document.body.appendChild(tab);
                htmlTabs.push({ element: tab, mesh: s });
            });

            scene.add(new THREE.AmbientLight(0xffffff, 0.9)); 
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
            dirLight.position.set(5, 10, 5);
            scene.add(dirLight);

            camera.position.z = 16; camera.position.y = 3;

            // 5. RAYCASTER LOGIC (HOVER AND CLICK)
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            window.addEventListener('mousemove', (event) => {
                mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
                mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);

                if (intersects.length > 0) {
                    document.body.style.cursor = 'pointer';
                    intersects[0].object.scale.set(1.4, 1.4, 1.4); 
                    controls.autoRotateSpeed = 0.1; 
                } else {
                    document.body.style.cursor = 'default';
                    agents.forEach(a => a.scale.set(1, 1, 1)); 
                    controls.autoRotateSpeed = 0.5;
                }
            });

            // FIXED NAVIGATION LOGIC 
            window.addEventListener('click', (event) => {
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);

                if (intersects.length > 0) {
                    const targetUrl = intersects[0].object.userData.url;
                    try {
                        // Attempt to alter the parent URL query string directly
                        window.parent.location.search = targetUrl;
                    } catch (e) {
                        // Fallback: create an anchor tag targeting the parent frame 
                        // to bypass iframe restrictions.
                        const link = document.createElement('a');
                        link.href = targetUrl;
                        link.target = "_parent"; 
                        document.body.appendChild(link);
                        link.click();
                    }
                }
            });

            window.addEventListener('resize', onWindowResize, false);
            function onWindowResize() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }

            const clock = new THREE.Clock();
            function animate() { 
                requestAnimationFrame(animate); 
                const elapsedTime = clock.getElapsedTime();
                
                particleSystem.rotation.y = elapsedTime * 0.02;
                mascotSystem.rotation.y = elapsedTime * 0.035; // Dogs orbit slightly faster
                
                agents.forEach((agent, i) => {
                    agent.position.y += Math.sin(elapsedTime * 2 + agent.userData.angle) * 0.015;

                    const vector = agent.position.clone();
                    vector.project(camera);
                    
                    const x = (vector.x * .5 + .5) * window.innerWidth;
                    const y = (vector.y * -.5 + .5) * window.innerHeight;
                    
                    const tabElement = htmlTabs[i].element;
                    if (vector.z > 1) {
                        tabElement.style.display = 'none';
                    } else {
                        tabElement.style.display = 'block';
                        tabElement.style.left = x + 'px';
                        tabElement.style.top = (y - 30) + 'px';
                        
                        if (agent.scale.x > 1.1) {
                            tabElement.style.opacity = '1';
                            tabElement.style.transform = 'translate(-50%, -150%) scale(1.1)';
                        } else {
                            tabElement.style.opacity = '0.8';
                            tabElement.style.transform = 'translate(-50%, -150%) scale(1)';
                        }
                    }
                });

                controls.update(); 
                renderer.render(scene, camera); 
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_cmu_galaxy, height=850)

# ======================= PAGE 1: MASTER DATA EXPLORER =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Tab 1: Master Data Explorer", description="View and filter all 12 raw data files.", color_name="yellow-70")
    
    selected_file = st.selectbox("Select a Dataset to Explore", ALL_FILES)
    df = load_raw_file(selected_file)
    
    if not df.empty and 'Error' not in df.columns:
        st.metric("Total Rows", len(df))
        filtered_df = dataframe_explorer(df, case=False)
        st.dataframe(filtered_df, use_container_width=True, height=500)
    else:
        st.error(df['Error'].iloc[0] if 'Error' in df.columns else "File is empty.")

# ======================= PAGE 2: DATA CLEANER =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Tab 2: Data Cleaner & Anomaly Resolution", description="Automated pipeline fixes applied based on Data Quality Assessment.", color_name="orange-70")
    
    c1, c2, c3 = st.columns(3)
    c1.info("**Fix 1: Index Mapping Extraction**\n\nParsed 'Landing Page' URLs to extract missing `utm_campaign` tags.")
    c2.info("**Fix 2: GA Deduplication & Melting**\n\nDropped duplicate `.1` headers and melted wide TimeSeries to long format.")
    c3.info("**Fix 3: GAds & LinkedIn Formatting**\n\nRegex stripped '%/,' symbols and dropped >80% empty LinkedIn columns.")
    
    st.subheader("Final Cleaned Master Hub Preview")
    if not master_df.empty:
        display_cols = [c for c in ['Unique_Campaign_ID', 'Category', 'Total_Combined_Spend', 'Total_Website_Users', 'CPWU'] if c in master_df.columns]
        st.dataframe(master_df[display_cols].sort_values(by='CPWU', ascending=True).head(50), use_container_width=True)
    else:
        st.warning("Master Dataframe is empty or dependencies failed to load.")

# ======================= PAGE 3: DATA ANALYSIS =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Tab 3: Statistical Data Analysis", description="Regression models and Hypothesis testing.", color_name="green-70")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Time-Series Regression")
        if not melted_ga.empty:
            top_camps = melted_ga.groupby('Session campaign')['User_Count'].sum().nlargest(5).index
            fig_trend = px.scatter(
                melted_ga[melted_ga['Session campaign'].isin(top_camps)], 
                x="Day_Number", y="User_Count", color="Session campaign", trendline="ols", template="plotly_white"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No TimeSeries data available for regression.")
            
    with colB:
        st.subheader("A/B Hypothesis Testing (T-Test)")
        st.markdown("Comparing Average Engagement Rate: **Google Ads vs LinkedIn Ads**")
        if 'Session source' in ga_utm.columns and 'Engagement rate' in ga_utm.columns:
            ga_utm['Engagement rate'] = pd.to_numeric(ga_utm['Engagement rate'], errors='coerce').fillna(0)
            google_eng = ga_utm[ga_utm['Session source'].str.contains('google', na=False, case=False)]['Engagement rate']
            li_eng = ga_utm[ga_utm['Session source'].str.contains('linkedin', na=False, case=False)]['Engagement rate']
            
            if len(google_eng) > 0 and len(li_eng) > 0:
                t_stat, p_val = stats.ttest_ind(google_eng, li_eng, equal_var=False)
                style_metric_cards(background_color="#FFFFFF", border_left_color="#44BBA4")
                mc1, mc2 = st.columns(2)
                mc1.metric("T-Statistic", f"{t_stat:.4f}")
                mc2.metric("P-Value", f"{p_val:.4e}")
                
                if p_val < 0.05:
                    st.success("Result: Statistically Significant (p < 0.05).")
                else:
                    st.warning("Result: Not Statistically Significant.")
            else:
                st.info("Insufficient platform split data for T-Test.")

# ======================= PAGE 4: INTERACTIVE DASHBOARD =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Tab 4: Interactive Dashboard", description="Cross-platform financial and operational insights.", color_name="light-blue-70")
    
    if not master_df.empty:
        mc1, mc2, mc3 = st.columns(3)
        total_spend = master_df['Total_Combined_Spend'].sum()
        total_users = master_df['Total_Website_Users'].sum()
        avg_cpwu = total_spend / total_users if total_users > 0 else 0
        
        mc1.metric("Total Spend Logged", f"${total_spend:,.2f}")
        mc2.metric("Total Users Acquired", f"{total_users:,.0f}")
        mc3.metric("Global Cost Per User", f"${avg_cpwu:.2f}")
        style_metric_cards(background_color="#FFFFFF", border_left_color="#00A6D6")
        
        st.divider()
        st.subheader("Cross-Platform Campaign ROI Matrix")
        valid_master = master_df[(master_df['Total_Website_Users'] > 0) & (master_df['Total_Combined_Spend'] > 0)]
        
        if not valid_master.empty:
            fig_roi = px.scatter(
                valid_master, x="Total_Combined_Spend", y="Total_Website_Users", 
                size="Average_Engagement_Rate" if 'Average_Engagement_Rate' in valid_master.columns else None, 
                color='Category' if 'Category' in valid_master.columns else None,
                hover_name="Unique_Campaign_ID" if 'Unique_Campaign_ID' in valid_master.columns else None,
                labels={"Total_Combined_Spend": "Ad Spend ($)", "Total_Website_Users": "Website Users"},
                template="plotly_white", height=600
            )
            fig_roi.add_hline(y=valid_master['Total_Website_Users'].mean(), line_dash="dot", annotation_text="Avg Users")
            fig_roi.add_vline(x=valid_master['Total_Combined_Spend'].mean(), line_dash="dot", annotation_text="Avg Spend")
            st.plotly_chart(fig_roi, use_container_width=True)
        else:
            st.info("Not enough numeric intersection data to plot ROI matrix.")
    else:
        st.warning("Master Dataframe is empty.")

# ======================= PAGE 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Tab 5: Knowledge Graph", description="Network topology of Campaign Structure, Vendors, and Mediums.", color_name="violet-70")
    
    if not master_df.empty and 'Unique_Campaign_ID' in master_df.columns:
        G = nx.Graph()
        
        for _, row in master_df.dropna(subset=['Unique_Campaign_ID', 'Category', 'Vendor']).head(50).iterrows():
            campaign = str(row['Unique_Campaign_ID'])
            category = str(row['Category'])
            vendor = str(row['Vendor'])
            
            G.add_node(category, title="Category", group=1, size=30, color="#E2C044")
            G.add_node(campaign, title="Campaign", group=2, size=20, color="#C41230")
            G.add_node(vendor, title="Vendor", group=3, size=25, color="#00A6D6")
            
            G.add_edge(category, campaign)
            G.add_edge(campaign, vendor)

        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True)
        net.from_nx(G)
        net.repulsion(node_distance=150, spring_length=150)
        
        path = os.path.join(tempfile.gettempdir(), 'knowledge_graph.html')
        net.save_graph(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=650)
    else:
        st.warning("Insufficient categorical data loaded to generate Knowledge Graph.")
