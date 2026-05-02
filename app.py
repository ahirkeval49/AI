import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CMU Campaign Intelligence", layout="wide", initial_sidebar_state="expanded")

# Simulated Custom Component Placeholder for 3D Hero
st.markdown("""
    <div style='background-color: #000; padding: 40px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;'>
        <h1 style='color: #C41230;'>CMU Campaign Intelligence</h1>
        <p><i>[ 3D Particle System Canvas Placeholder (WebGL/Three.js) ]</i></p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA LOADING & CLEANING FUNCTIONS (Cached for performance)
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        # Load Raw Data from the 'data/' folder
        index_df = pd.read_csv('data/UCM Campaign Index.csv')
        campaign_mgmt = pd.read_csv('data/2024-25_Campaign_Management_1769521985.csv')
        ga_time = pd.read_csv('data/GA_FY25_TimeSeries (1).csv')
        ga_utm = pd.read_csv('data/GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', skiprows=1) # Skipping header row if malformed
        gads_perf = pd.read_csv('data/GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv')
        linkedin_perf = pd.read_csv('data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv')

        # --- Cleaning Anomaly 1: Structural Bloat in Campaign Mgmt ---
        campaign_clean = campaign_mgmt.dropna(how='all')
        if 'Name' in campaign_clean.columns:
            campaign_clean = campaign_clean.dropna(subset=['Name'])

        # --- Cleaning Anomaly 2: Wide-to-Long Reshaping for TimeSeries ---
        melted_ga = pd.melt(ga_time, id_vars=['Session campaign', 'Segment'], var_name='Day', value_name='User_Count')
        melted_ga['Day_Number'] = melted_ga['Day'].str.extract('(\d+)').astype(float)
        melted_ga['User_Count'] = pd.to_numeric(melted_ga['User_Count'], errors='coerce').fillna(0)

        # --- Cleaning Anomaly 3: String/Numeric Type Mismatches in Google Ads ---
        gads_perf.replace('--', np.nan, inplace=True)
        cols_to_clean = ['Clicks', 'Impr.', 'CTR', 'Cost']
        for col in cols_to_clean:
            if col in gads_perf.columns:
                gads_perf[col] = gads_perf[col].astype(str).str.replace(r'[,\%\$]', '', regex=True)
                gads_perf[col] = pd.to_numeric(gads_perf[col], errors='coerce')

        # --- Cleaning Anomaly 4: Dimensionality Reduction in LinkedIn ---
        threshold = len(linkedin_perf) * 0.8
        linkedin_clean = linkedin_perf.dropna(thresh=len(linkedin_perf) - threshold, axis=1)
        
        return index_df, campaign_clean, melted_ga, ga_utm, gads_perf, linkedin_clean
        
    except FileNotFoundError as e:
        st.error(f"File missing: {e}. Please ensure all CSVs are in the 'data/' folder.")
        st.stop()

# Load the data
index_df, campaign_clean, melted_ga, ga_utm, gads_perf, linkedin_clean = load_and_clean_data()

# ---------------------------------------------------------
# 3. UNIFIED DATA MODEL (Master View Merge)
# ---------------------------------------------------------
@st.cache_data
def create_master_view(index_df, ga_utm, gads_perf, linkedin_clean):
    # Standardize join keys
    if 'UTM campaign' in index_df.columns:
        index_df['utm_clean'] = index_df['UTM campaign'].astype(str).str.lower().str.strip()
    
    # Process GA UTM
    ga_utm_renamed = ga_utm.copy()
    if 'Session campaign' in ga_utm_renamed.columns:
        ga_utm_renamed['utm_clean'] = ga_utm_renamed['Session campaign'].astype(str).str.lower().str.strip()
        ga_agg = ga_utm_renamed.groupby('utm_clean').agg(
            Total_Website_Users=('Total users', 'sum'),
            Average_Engagement_Rate=('Engagement rate', 'mean')
        ).reset_index()
    else:
        ga_agg = pd.DataFrame(columns=['utm_clean', 'Total_Website_Users', 'Average_Engagement_Rate'])

    # Mock Spend Data based on files (Replace 'Cost'/'Total Spend' with actual column names if they differ slightly)
    # Grouping Google Ads Spend
    if 'Ad name' in gads_perf.columns and 'Cost' in gads_perf.columns:
        gads_perf['utm_clean'] = gads_perf['Ad name'].astype(str).str.lower().str.strip()
        gads_agg = gads_perf.groupby('utm_clean').agg(Total_GAds_Spend=('Cost', 'sum')).reset_index()
    else:
        gads_agg = pd.DataFrame(columns=['utm_clean', 'Total_GAds_Spend'])

    # Grouping LinkedIn Spend
    if 'Campaign Name' in linkedin_clean.columns and 'Total Spend' in linkedin_clean.columns:
        linkedin_clean['utm_clean'] = linkedin_clean['Campaign Name'].astype(str).str.lower().str.strip()
        li_agg = linkedin_clean.groupby('utm_clean').agg(Total_LinkedIn_Spend=('Total Spend', 'sum')).reset_index()
    else:
        li_agg = pd.DataFrame(columns=['utm_clean', 'Total_LinkedIn_Spend'])

    # Merge everything to the Index Hub
    master_df = index_df.copy()
    master_df = pd.merge(master_df, ga_agg, on='utm_clean', how='left')
    master_df = pd.merge(master_df, gads_agg, on='utm_clean', how='left')
    master_df = pd.merge(master_df, li_agg, on='utm_clean', how='left')

    # Fill NA and calculate Cost Per Website User
    master_df.fillna({'Total_GAds_Spend': 0, 'Total_LinkedIn_Spend': 0, 'Total_Website_Users': 0}, inplace=True)
    master_df['Total_Combined_Spend'] = master_df['Total_GAds_Spend'] + master_df['Total_LinkedIn_Spend']
    master_df['CPWU'] = np.where(master_df['Total_Website_Users'] > 0, 
                                 master_df['Total_Combined_Spend'] / master_df['Total_Website_Users'], 0)
    
    return master_df

master_df = create_master_view(index_df, ga_utm, gads_perf, linkedin_clean)

# ---------------------------------------------------------
# 4. STREAMLIT UI LAYOUT & TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Walkthrough & KPIs", "📈 Statistical Analysis", "🎯 Master View (ROI Matrix)"])

# --- TAB 1: Data Walkthrough & Top-line KPIs ---
with tab1:
    st.header("Platform Highlights & Cleaned Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Logged Campaigns", value=len(index_df))
    with col2:
        st.metric(label="Total Cross-Platform Spend", value=f"${master_df['Total_Combined_Spend'].sum():,.2f}")
    with col3:
        st.metric(label="Total Website Users Acquired", value=f"{master_df['Total_Website_Users'].sum():,.0f}")
        
    st.divider()
    
    st.subheader("Data Cleaning Walkthrough")
    with st.expander("View Cleaned LinkedIn Ads Data (Sparsity Handled)"):
        st.markdown("Removed columns with > 80% null values to isolate core performance metrics.")
        st.dataframe(linkedin_clean.head(100))
        
    with st.expander("View Reshaped GA TimeSeries Data (Wide to Long)"):
        st.markdown("Melted 365 daily columns into rows to enable proper temporal analysis.")
        st.dataframe(melted_ga.head(100))

# --- TAB 2: Statistical Analysis ---
with tab2:
    st.header("Advanced Statistical Analysis")
    
    st.subheader("1. Time-Series Analysis (OLS Regression)")
    st.markdown("Evaluating the long-term trend of user acquisition over the campaign lifecycle.")
    
    # Plotly Trendline (Requires statsmodels)
    if not melted_ga.empty:
        # Filter down to top 5 campaigns for visual clarity
        top_campaigns = melted_ga.groupby('Session campaign')['User_Count'].sum().nlargest(5).index
        filtered_ga = melted_ga[melted_ga['Session campaign'].isin(top_campaigns)]
        
        fig1 = px.scatter(
            filtered_ga, x="Day_Number", y="User_Count", color="Session campaign",
            trendline="ols", title="Top 5 Campaigns: Daily Traffic with OLS Regression"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    st.divider()
    
    st.subheader("2. A/B Hypothesis Testing (Independent T-Test)")
    st.markdown("Testing if there is a statistically significant difference in engagement between Google vs. LinkedIn campaigns.")
    
    # Perform T-Test on Engagement Rate (Mock logic based on UTM source)
    if 'Session source' in ga_utm.columns and 'Engagement rate' in ga_utm.columns:
        ga_utm['Engagement rate'] = pd.to_numeric(ga_utm['Engagement rate'], errors='coerce').fillna(0)
        google_engagement = ga_utm[ga_utm['Session source'].str.contains('google', na=False, case=False)]['Engagement rate']
        linkedin_engagement = ga_utm[ga_utm['Session source'].str.contains('linkedin', na=False, case=False)]['Engagement rate']
        
        if len(google_engagement) > 0 and len(linkedin_engagement) > 0:
            t_stat, p_val = stats.ttest_ind(google_engagement, linkedin_engagement, equal_var=False)
            
            st.write(f"**T-Statistic:** {t_stat:.4f}")
            st.write(f"**P-Value:** {p_val:.4e}")
            
            if p_val < 0.05:
                st.success(f"Result: The difference in engagement rates is statistically significant (p < 0.05).")
            else:
                st.warning(f"Result: No statistically significant difference in engagement rates (p >= 0.05).")

# --- TAB 3: The Unified Master View ---
with tab3:
    st.header("Unified Cross-Platform ROI")
    st.markdown("""
    This model merges Google Ads spend, LinkedIn Ads spend, and GA website traffic against the central Campaign Index.
    Use the **Cost Per Website User (CPWU)** to find efficiency.
    """)
    
    # Clean up the display dataframe
    display_cols = ['Unique_Campaign_ID', 'Category', 'Total_Combined_Spend', 'Total_Website_Users', 'CPWU', 'Average_Engagement_Rate']
    available_cols = [c for c in display_cols if c in master_df.columns]
    
    st.dataframe(master_df[available_cols].sort_values(by='CPWU', ascending=True).head(50))
    
    st.divider()
    
    st.subheader("Campaign Efficiency Matrix")
    st.markdown("Ideal campaigns live in the **Top-Left quadrant** (High Users, Low Spend).")
    
    valid_master = master_df[(master_df['Total_Website_Users'] > 0) & (master_df['Total_Combined_Spend'] > 0)]
    
    if not valid_master.empty:
        # Determine coloring variable based on available columns
        color_col = 'Category' if 'Category' in valid_master.columns else None
            
        fig2 = px.scatter(
            valid_master, 
            x="Total_Combined_Spend", 
            y="Total_Website_Users", 
            size="Average_Engagement_Rate", 
            color=color_col,
            hover_name="Unique_Campaign_ID" if 'Unique_Campaign_ID' in valid_master.columns else None,
            labels={
                "Total_Combined_Spend": "Total Ad Spend ($)",
                "Total_Website_Users": "Website Users Acquired"
            },
            template="plotly_dark"
        )
        # Add Crosshairs for average lines
        fig2.add_hline(y=valid_master['Total_Website_Users'].mean(), line_dash="dot", annotation_text="Avg Users", annotation_position="top left")
        fig2.add_vline(x=valid_master['Total_Combined_Spend'].mean(), line_dash="dot", annotation_text="Avg Spend", annotation_position="top right")
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough overlapping spend/user data to generate the ROI matrix. Check mapping keys.")
