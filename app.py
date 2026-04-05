# ==========================================
# ACTUAL DATA PIPELINE (ETL)
# ==========================================
@st.cache_data
def load_and_process_real_data():
    """
    Reads the actual CSVs from the /data directory, cleans strings, 
    melts time series, and joins GA4 to Google Ads.
    """
    data_dir = "data/"
    
    if not os.path.exists(data_dir + "GAds_FY26_Totals_Jul-Dec2025.csv"):
        st.error(f"Waiting for files... Please ensure your CSVs are placed in a folder named '{data_dir}' relative to this script.")
        st.stop()

    # -----------------------------------------------------
    # 1. ENGINEER MASTER DATA (Join GAds + GA4 UTMs)
    # -----------------------------------------------------
    # Load Google Ads Totals
    df_gads = pd.read_csv(data_dir + "GAds_FY26_Totals_Jul-Dec2025.csv")
    
    # Clean anomalies: Remove summary rows
    df_gads = df_gads[~df_gads['Campaign'].astype(str).str.contains('Total', na=False)]
    
    # --- 🚨 FIX FOR MISSING COST COLUMN 🚨 ---
    # If the export omitted the 'Cost' column, we calculate it dynamically
    if 'Cost' not in df_gads.columns:
        df_gads['Clicks'] = df_gads['Clicks'].astype(str).str.replace(',', '').str.replace('--', '0').astype(float)
        df_gads['Avg. CPC'] = df_gads['Avg. CPC'].astype(str).str.replace(',', '').str.replace('--', '0').astype(float)
        df_gads['Cost'] = df_gads['Clicks'] * df_gads['Avg. CPC']
    else:
        df_gads['Cost'] = df_gads['Cost'].astype(str).str.replace(',', '').str.replace('--', '0').astype(float)
    # -----------------------------------------

    # Clean remaining columns
    df_gads['Impr.'] = df_gads['Impr.'].astype(str).str.replace(',', '').str.replace('--', '0').astype(float)
    df_gads['CTR_Clean'] = df_gads['CTR'].astype(str).str.rstrip('%').str.replace('--', '0').astype(float) / 100
    
    # Load GA4 Web Totals
    df_ga4 = pd.read_csv(data_dir + "GA_FY26_UTM_Totals_Jul-Dec2025.csv")
    df_ga4['Average session duration'] = pd.to_numeric(df_ga4['Average session duration'], errors='coerce')
    
    # THE ROSETTA STONE: Merge Google Ads Spend with GA4 Web Session Data
    df_master = pd.merge(
        df_gads[['Campaign', 'Cost', 'Impr.', 'CTR_Clean']], 
        df_ga4[['Session campaign', 'Average session duration']], 
        left_on='Campaign', 
        right_on='Session campaign', 
        how='inner'
    )
    df_master = df_master.dropna(subset=['Average session duration', 'Cost'])

    # -----------------------------------------------------
    # 2. ENGINEER AUDIENCE CLUSTERING
    # -----------------------------------------------------
    df_aud_raw = pd.read_csv(data_dir + "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv")
    
    df_aud = df_aud_raw[~df_aud_raw['Audience segment'].isin(['People not in audiences', '(not set)'])]
    df_aud['CTR'] = df_aud['CTR'].astype(str).str.rstrip('%').str.replace('--', 'NaN').astype(float) / 100
    df_aud['TrueView view rate'] = df_aud['TrueView view rate'].astype(str).str.rstrip('%').str.replace('--', 'NaN').astype(float) / 100
    
    df_aud = df_aud.dropna(subset=['CTR', 'TrueView view rate'])
    
    if len(df_aud) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(df_aud[['CTR', 'TrueView view rate']])
        df_aud['Persona'] = kmeans.labels_
        persona_map = {0: "Cultural Generalist", 1: "High-Intent Specialist", 2: "Knowledge Seeker"}
        df_aud['Persona'] = df_aud['Persona'].map(persona_map)

    # -----------------------------------------------------
    # 3. MELT TIME SERIES
    # -----------------------------------------------------
    df_ts_raw = pd.read_csv(data_dir + "GA_FY26_TimeSeries.csv")
    
    day_cols = [c for c in df_ts_raw.columns if 'Total users_Day' in c]
    
    df_ts = pd.melt(df_ts_raw, id_vars=['Session campaign'], value_vars=day_cols, 
                    var_name='Day_Raw', value_name='Users')
    
    df_ts['Day'] = df_ts['Day_Raw'].str.replace('Total users_Day', '').astype(int)
    
    top_campaigns = df_ts.groupby('Session campaign')['Users'].sum().nlargest(2).index
    df_ts = df_ts[df_ts['Session campaign'].isin(top_campaigns)]

    return df_master, df_aud, df_ts
