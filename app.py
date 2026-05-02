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

# Routing Logic
if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# ---------------------------------------------------------
# 2. GLOBAL CUSTOM UI: CENTER-BOTTOM RADIAL MENU
# ---------------------------------------------------------
# PERSISTENCE: Menu stays open once clicked
menu_checked_state = "checked" 

radial_menu_html = f"""
<style>
.radial-nav {{ position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); z-index: 9999999; }}
#menu-toggle {{ display: none; }}
.menu-button {{ width: 80px; height: 80px; background-color: #C41230; color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 32px; cursor: pointer; box-shadow: 0 0 30px rgba(196, 18, 48, 0.6); position: relative; z-index: 20; border: 3px solid #FFF; font-family: sans-serif; transition: transform 0.3s ease; }}
#menu-toggle:checked ~ .menu-button {{ transform: rotate(45deg); background-color: #222; }}
.menu-item {{ position: absolute; top: 15px; left: 15px; width: 60px; height: 60px; background-color: #333; color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 10px; text-decoration: none; font-weight: bold; font-family: sans-serif; opacity: 0; transform: scale(0); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); box-shadow: 0 4px 15px rgba(0,0,0,0.6); border: 1px solid #555; z-index: 10; text-align: center; color: white !important; }}
.menu-item:hover {{ background-color: #C41230 !important; border-color: white !important; transform: scale(1.15) !important; }}
/* Radial fan positions */
#menu-toggle:checked ~ .item-1 {{ opacity: 1; transform: translate(-180px, -40px) scale(1); }}
#menu-toggle:checked ~ .item-2 {{ opacity: 1; transform: translate(-110px, -120px) scale(1); }}
#menu-toggle:checked ~ .item-3 {{ opacity: 1; transform: translate(0px, -160px) scale(1); }}
#menu-toggle:checked ~ .item-4 {{ opacity: 1; transform: translate(110px, -120px) scale(1); }}
#menu-toggle:checked ~ .item-5 {{ opacity: 1; transform: translate(180px, -40px) scale(1); }}
#menu-toggle:checked ~ .item-6 {{ opacity: 1; transform: translate(0px, 80px) scale(1); }}
.label-text {{ position: absolute; bottom: -35px; left: 50%; transform: translateX(-50%); width: 300px; color: #888; font-family: sans-serif; font-size: 14px; font-weight: bold; text-align: center; pointer-events: none; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }}
</style>
<div class="radial-nav">
<input type="checkbox" id="menu-toggle" {menu_checked_state}>
<label for="menu-toggle" class="menu-button">✦</label>
<div class="label-text">NAVIGATE DATA TABS</div>
<a href="?page=home" target="_top" class="menu-item item-1">HOME</a>
<a href="?page=explorer" target="_top" class="menu-item item-2">EXPLORE</a>
<a href="?page=cleaner" target="_top" class="menu-item item-3">CLEAN</a>
<a href="?page=analysis" target="_top" class="menu-item item-4">STATS</a>
<a href="?page=dashboard" target="_top" class="menu-item item-5">DASH</a>
<a href="?page=graph" target="_top" class="menu-item item-6">GRAPH</a>
</div>
"""
st.markdown(radial_menu_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. DATA LOADING & CLEANING (DQA Implementation)
# ---------------------------------------------------------
@st.cache_data
def load_raw_file(filename):
    try:
        skip = 1 if 'UTM_Totals' in filename else 0
        df = pd.read_csv(f'data/{filename}', skiprows=skip)
        for c in [col for col in df.columns if col.endswith('.1')]:
            df.drop(columns=c, inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

@st.cache_data
def build_master_models():
    # Loading logic (Combine FY25 & FY26)
    idx = load_raw_file("UCM Campaign Index.csv")
    ga_utm = pd.concat([load_raw_file("GA_FY25_UTM_Totals_Jul2024-Jun2025.csv"), load_raw_file("GA_FY26_UTM_Totals_Jul-Dec2025.csv")])
    ga_time = pd.concat([load_raw_file("GA_FY25_TimeSeries (1).csv"), load_raw_file("GA_FY26_TimeSeries.csv")])
    gads = load_raw_file("GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv")
    li = load_raw_file("LinkedIn_Ad_Performance_Feb2024_Dec2025.csv")
    
    # Cleaning... (Same as previous expert pipelines)
    # Melting wide-to-long, regex stripping symbols, dropping null-columns
    return idx, ga_utm, ga_time, gads, li

# Mock data build for rendering
ALL_FILES = ["UCM Campaign Index.csv", "GA_FY25_TimeSeries (1).csv", "GAds_FY25_Totals_Jul2024-Jun2025.csv", "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv"]
master_df = pd.DataFrame({"Unique_Campaign_ID":["A1","A2"],"Category":["Brand","Brand"],"Total_Combined_Spend":[1000,2000],"Total_Website_Users":[500,1000],"CPWU":[2,2]})

# ---------------------------------------------------------
# 4. VIEW RENDERING: 3D HOME WITH GALAXY PARTICLES
# ---------------------------------------------------------
if current_page == "home":
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 800;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    
    three_js_galaxy = """
    <!DOCTYPE html>
    <html>
    <head>
        <style> body { margin: 0; overflow: hidden; background-color: #050505; } </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(60, window.innerWidth / 700, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({antialias: true});
            renderer.setSize(window.innerWidth, 700);
            document.body.appendChild(renderer.domElement);
            
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.autoRotate = true; controls.autoRotateSpeed = 0.8;

            // 1. Galaxy: 15,000+ tiny particles
            const pCount = 15000;
            const pGeo = new THREE.BufferGeometry();
            const pos = new Float32Array(pCount * 3);
            for(let i=0; i<pCount*3; i++) pos[i] = (Math.random()-0.5)*30;
            pGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            const pMat = new THREE.PointsMaterial({size: 0.015, color: 0xffffff, transparent: true, opacity: 0.6});
            scene.add(new THREE.Points(pGeo, pMat));

            // 2. CMU Volumetric Core
            const bGeo = new THREE.BoxGeometry(0.4, 0.4, 0.4);
            const bMat = new THREE.MeshStandardMaterial({color: 0xC41230, emissive: 0xC41230, emissiveIntensity: 0.2});
            const core = new THREE.Group();
            const coords = [[0,0],[1,0],[2,0],[3,0],[0,-1],[0,-2],[0,-3],[0,-4],[1,-4],[2,-4],[3,-4],[5,0],[9,0],[5,-1],[6,-1],[8,-1],[9,-1],[5,-2],[7,-2],[9,-2],[5,-3],[9,-3],[5,-4],[9,-4],[11,0],[14,0],[11,-1],[14,-1],[11,-2],[14,-2],[11,-3],[14,-3],[11,-4],[12,-4],[13,-4],[14,-4]];
            coords.forEach(p => {
                const m = new THREE.Mesh(bGeo, bMat);
                m.position.set(p[0]*0.5 - 3.5, p[1]*0.5 + 1, 0);
                core.add(m);
            });
            scene.add(core);

            // 3. Orbiting Nodes (Agents)
            const agents = [];
            const nodes = [0xE2C044, 0xE87A5D, 0x44BBA4, 0x00A6D6, 0x9B5DE5];
            nodes.forEach((c, i) => {
                const s = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 32), new THREE.MeshStandardMaterial({color: c, emissive: c, emissiveIntensity: 0.5}));
                const a = (i/5)*Math.PI*2;
                s.position.set(Math.cos(a)*7, Math.sin(a)*2, Math.sin(a)*7);
                scene.add(s);
            });

            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            camera.position.z = 15;

            function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_galaxy, height=750)

# Other pages (Explorer, Cleaner, ROI, etc.)
elif current_page == "explorer":
    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)
    colored_header(label="Master Data Explorer", description="Deep dive into the 12-file ecosystem.", color_name="red-70")
    st.dataframe(master_df)

elif current_page == "graph":
    st.markdown("<div style='height:80px;'></div>", unsafe_allow_html=True)
    colored_header(label="Knowledge Graph", description="Network topology of campaign intelligence.", color_name="violet-70")
    st.write("Generating graph...")
