"""
============================================================================
INTERACTIVE SPATIAL DASHBOARD FOR DENGUE FEVER (DBD) IN INDONESIA
STREAMLIT VERSION - FINAL PERFECTED (HYBRID INTERPOLATION)
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN, Queen, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error
import warnings
from datetime import datetime
import base64
from io import BytesIO
import json
from scipy.stats import norm
from shapely.geometry import Point

# Import additional libraries for interpolation
try:
    from scipy.interpolate import griddata
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Indonesia DBD Risk Explorer",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_disease_data():
    """Load and preprocess disease data with auto-column detection"""
    try:
        # Load data
        data = pd.read_csv("DATA SPASIAL EPIDEM - VARIABEL DEPENDEN (1).csv")
        
        # 1. Hapus kolom Rabies jika ada
        if 'Rabies' in data.columns:
            data.drop(columns=['Rabies'], inplace=True)
            
        # 2. Cek jumlah kolom untuk penamaan otomatis
        if len(data.columns) == 9:
            # Data Asli (Lengkap)
            data.columns = [
                "province", "malaria", "dbd", "filariasis",
                "sanitation", "pop_density", "hospitals", "poor_pct", "population"
            ]
        elif len(data.columns) == 7:
            # Data Khusus DBD (Asumsi urutan kolom)
            data.columns = [
                "province", "dbd",
                "sanitation", "pop_density", "hospitals", "poor_pct", "population"
            ]
        else:
            # Fallback
            st.warning(f"Jumlah kolom tidak standar ({len(data.columns)}). Menggunakan nama asli.")
            data.columns = [c.lower() for c in data.columns]
            if 'provinsi' in data.columns: data.rename(columns={'provinsi': 'province'}, inplace=True)
            if 'jumlah_penduduk' in data.columns: data.rename(columns={'jumlah_penduduk': 'population'}, inplace=True)

        # 3. Pembersihan Data (Numeric Conversion)
        numeric_cols = ["dbd", "sanitation", "pop_density", "hospitals", "poor_pct", "population"]
        for extra in ['malaria', 'filariasis']:
            if extra in data.columns: numeric_cols.append(extra)
        
        for col in numeric_cols:
            if col in data.columns:
                if data[col].dtype == object:
                    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')
                data[col].fillna(data[col].median(), inplace=True)
        
        # 4. Hitung Insidensi DBD
        data['population_actual'] = data['population'] * 1000
        selected_disease = 'dbd'
        
        data[f'{selected_disease}_prev'] = (data[selected_disease] / data['population_actual']) * 100000
        data[f'{selected_disease}_prev'].replace([np.inf, -np.inf], 0, inplace=True)
        data[f'{selected_disease}_prev'].fillna(0, inplace=True)
        
        # Hitung kuartil
        batas25 = data[f'{selected_disease}_prev'].quantile(0.25)
        batas50 = data[f'{selected_disease}_prev'].quantile(0.50)
        batas75 = data[f'{selected_disease}_prev'].quantile(0.75)

        data[f'{selected_disease}_risk'] = data[f'{selected_disease}_prev'].apply(
            lambda x: get_risk_level(x, selected_disease, batas25, batas50, batas75)
        )
        
        return data
    
    except Exception as e:
        st.error(f"Error loading disease data: {e}")
        return None


@st.cache_data
def load_spatial_data():
    """Load spatial boundary data"""
    try:
        gdf = gpd.read_file('gadm41_IDN_1.json')
        return gdf
    except Exception as e:
        st.error(f"Error loading spatial data: {e}")
        return None

@st.cache_data
def merge_data(disease_data, _spatial_data):
    """Merge disease and spatial data"""
    try:
        merged = _spatial_data.merge( 
            disease_data, 
            left_on='NAME_1', 
            right_on='province', 
            how='left'
        )
        merged = merged.dropna(subset=['dbd'])
        return merged
    except Exception as e:
        st.error(f"Error merging data: {e}")
        return None

# ============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# ============================================================================

@st.cache_data
def create_weights(_gdf, k=3):
    """Create spatial weights matrix"""
    try:
        w = KNN.from_dataframe(_gdf, k=k)
        w.transform = 'r'
        return w
    except Exception as e:
        st.error(f"Error creating weights: {e}")
        return None

def calculate_morans_i(gdf, disease_var, w):
    """Calculate Global Moran's I"""
    try:
        prevalence_var = f"{disease_var}_prev"
        y = gdf[prevalence_var].values
        
        valid_idx = ~np.isnan(y)
        gdf_valid = gdf.loc[valid_idx].reset_index(drop=True)
        y = y[valid_idx]

        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        if len(y) < 3:
            return None
        
        moran = Moran(y, w_valid)
        
        return {
            'Statistic': moran.I,
            'P_value': moran.p_sim,
            'Expected': moran.EI,
            'Z_score': moran.z_sim
        }
    except Exception as e:
        st.error(f"Error in Moran's I: {e}")
        return None

def calculate_lisa(gdf, disease_var, w):
    """Calculate Local Moran's I (LISA)"""
    try:
        prevalence_var = f"{disease_var}_prev"
        y = gdf[prevalence_var].values
        
        valid_idx = ~np.isnan(y)
        y = y[valid_idx]
        gdf_valid = gdf[valid_idx].copy()

        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        if len(y) < 3:
            return None
        
        lisa = Moran_Local(y, w_valid)
        
        gdf_valid['Ii'] = lisa.Is
        gdf_valid['p_value'] = lisa.p_sim
        
        y_std = (y - y.mean()) / y.std()
        spatial_lag = lag_spatial(w_valid, y_std)
        
        gdf_valid['Cluster'] = 'Not Significant'
        sig_mask = gdf_valid['p_value'] < 0.05
        
        gdf_valid.loc[sig_mask & (y_std > 0) & (spatial_lag > 0), 'Cluster'] = 'HH (Hotspot)'
        gdf_valid.loc[sig_mask & (y_std < 0) & (spatial_lag < 0), 'Cluster'] = 'LL (Coldspot)'
        gdf_valid.loc[sig_mask & (y_std > 0) & (spatial_lag < 0), 'Cluster'] = 'HL (Outlier)'
        gdf_valid.loc[sig_mask & (y_std < 0) & (spatial_lag > 0), 'Cluster'] = 'LH (Outlier)'
        
        return gdf_valid
    except Exception as e:
        st.error(f"Error in LISA: {e}")
        return None

def fit_spatial_models(gdf, disease_var, x_vars, w):
    """Fit spatial econometric models"""
    try:
        prevalence_var = f"{disease_var}_prev"
        data = gdf[[prevalence_var] + x_vars].dropna()
        y = data[prevalence_var].values.reshape(-1, 1)
        X = data[x_vars].values
        
        if len(y) < 5: return None
        
        gdf_valid = gdf.loc[data.index]
        w_valid = Queen.from_dataframe(gdf_valid)
        w_valid.transform = 'r'
        
        models = {}
        try: models['ols'] = OLS(y, X, name_x=x_vars, name_y=prevalence_var)
        except: models['ols'] = None
        try: models['lag'] = ML_Lag(y, X, w_valid, name_x=x_vars, name_y=prevalence_var)
        except: models['lag'] = None
        try: models['error'] = ML_Error(y, X, w_valid, name_x=x_vars, name_y=prevalence_var)
        except: models['error'] = None
        return models
    except Exception as e:
        st.error(f"Error fitting models: {e}")
        return None

# ============================================================================
# NEW FUNCTIONS: EPI & INTERPOLATION (WITH HYBRID FILLING)
# ============================================================================

def calculate_epi_measures(df, disease_var, exposure_var):
    """Calculate Odds Ratio and Risk Ratio"""
    try:
        prev_var = f"{disease_var}_prev"
        median_exposure = df[exposure_var].median()
        median_outcome = df[prev_var].median()
        
        if exposure_var == 'sanitation':
             df['exposure_bin'] = np.where(df[exposure_var] < median_exposure, 1, 0)
        else: 
             df['exposure_bin'] = np.where(df[exposure_var] > median_exposure, 1, 0)
             
        df['outcome_bin'] = np.where(df[prev_var] > median_outcome, 1, 0)
        
        a = len(df[(df['exposure_bin']==1) & (df['outcome_bin']==1)])
        b = len(df[(df['exposure_bin']==1) & (df['outcome_bin']==0)])
        c = len(df[(df['exposure_bin']==0) & (df['outcome_bin']==1)])
        d = len(df[(df['exposure_bin']==0) & (df['outcome_bin']==0)])
        
        try: or_val = (a * d) / (b * c)
        except: or_val = np.nan
        try: rr_val = (a / (a + b)) / (c / (c + d))
        except: rr_val = np.nan
            
        return {'table': [[a, b], [c, d]], 'OR': or_val, 'RR': rr_val}
    except Exception: return None

@st.cache_data
def perform_idw_interpolation(_gdf, disease_var, resolution=100):
    """
    Perform Hybrid Interpolation (Linear + Nearest) AND clip to Indonesia.
    This fixes the 'cut off' issue in islands like Papua.
    """
    try:
        prev_var = f"{disease_var}_prev"
        valid_data = _gdf.dropna(subset=[prev_var, 'geometry'])
        
        if len(valid_data) < 4: return None, None, None

        valid_data['centroid'] = valid_data.geometry.centroid
        x = valid_data['centroid'].x
        y = valid_data['centroid'].y
        z = valid_data[prev_var]
        
        # 1. Create Grid
        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # 2. HYBRID INTERPOLATION
        # Step A: Linear interpolation (Smooth, but creates NaNs outside convex hull)
        zi_linear = griddata((x, y), z, (xi, yi), method='linear')
        
        # Step B: Nearest interpolation (Blocky, but fills EVERYTHING)
        zi_nearest = griddata((x, y), z, (xi, yi), method='nearest')
        
        # Step C: Fill NaNs in Linear with Nearest values
        # This fixes the "Papua cut off" problem!
        zi = np.where(np.isnan(zi_linear), zi_nearest, zi_linear)
        
        # 3. MASKING (Clipping to Map)
        grid_points = [Point(x_val, y_val) for x_val, y_val in zip(xi.flatten(), yi.flatten())]
        grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=_gdf.crs)
        
        simple_indo = _gdf.copy()
        simple_indo['geometry'] = simple_indo.geometry.simplify(0.05) 
        
        joined = gpd.sjoin(grid_gdf, simple_indo, how='left', predicate='intersects')
        # Remove duplicates from border points
        joined = joined[~joined.index.duplicated(keep='first')]
        
        mask = joined['index_right'].isna()
        zi_flat = zi.flatten()
        zi_flat[mask] = np.nan
        zi_masked = zi_flat.reshape(zi.shape)
        
        return xi, yi, zi_masked
        
    except Exception as e:
        st.error(f"Interpolation error: {e}")
        return None, None, None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_risk_level(value, disease_var, q1, q2, q3):
    thresh = {'low': q1, 'medium': q2, 'high': q3}
    if pd.isna(value): return "No Data"
    elif value < thresh['low']: return "Low Risk"
    elif value < thresh['medium']: return "Medium Risk"
    elif value < thresh['high']: return "High Risk"
    else: return "Very High Risk"

def get_risk_color(risk_level):
    colors = {"Low Risk": "#FFEB3B", "Medium Risk": "#FF9800", 
              "High Risk": "#F0592B", "Very High Risk": "#020000", "No Data": "#9E9E9E"}
    return colors.get(risk_level, "#0A0000")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🦟 Indonesia DBD Risk Explorer")
    
    st.sidebar.header("⚙ Settings")
    selected_disease = 'dbd'
    st.sidebar.info("Focus: Dengue Fever (DBD)")
    
    risk_factors = {
        'sanitation': 'Clean Water Access',
        'pop_density': 'Population Density',
        'hospitals': 'Hospital Availability',
        'poor_pct': 'Poverty Rate'
    }
    
    selected_factors = st.sidebar.multiselect(
        "Risk Factors:",
        options=list(risk_factors.keys()),
        default=list(risk_factors.keys()),
        format_func=lambda x: risk_factors[x]
    )
    
    with st.spinner("Loading data..."):
        disease_data = load_disease_data()  
        spatial_data = load_spatial_data()
        
        if disease_data is None or spatial_data is None:
            st.error("Failed to load data. Please check file paths.")
            return
        
        # Spatial Data Cleaning
        spatial_data.iloc[6, 3] = 'DKIJakarta'
        spatial_data.iloc[33, 3] = 'DaerahIstimewaYogyakarta'
        spatial_data.iloc[2, 3] = 'KepulauanBangkaBelitung'
        spatial_data.iloc[34:38, 3] = spatial_data.iloc[34:38, 11]
        spatial_data['NAME_1'] = spatial_data['NAME_1'].astype(str).str.replace(' ', '', regex=False)
        disease_data['province'] = disease_data['province'].astype(str).str.replace(' ', '', regex=False)

        merged_data = merge_data(disease_data, spatial_data)
        merged_data.reset_index(drop=True, inplace=True)
        
        if merged_data is None:
            st.error("Failed to merge data.")
            return
        
        weights = create_weights(merged_data)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺 Interactive Map",
        "🔍 Province Explorer",
        "📊 Risk Patterns",
        "📈 Statistical Model",
        "💊 Epidemiological Measures",
        "📋 Data Summary",
        "ℹ How to Use"
    ])
    
    # TAB 1: INTERACTIVE MAP
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        total_cases = merged_data[selected_disease].sum()
        risk_var = f"{selected_disease}_risk"
        prevalence_var = f"{selected_disease}_prev"
        
        with col1: st.metric("Total DBD Cases", f"{int(total_cases):,}")
        with col2: st.metric("High Risk Provinces", (merged_data[risk_var].isin(['High Risk', 'Very High Risk'])).sum())
        with col3: st.metric("Medium Risk Provinces", (merged_data[risk_var] == 'Medium Risk').sum())
        with col4: st.metric("Low Risk Provinces", (merged_data[risk_var] == 'Low Risk').sum())
        
        st.subheader('DBD Incidence Map')
        merged_data = merged_data.set_index("NAME_1")
        geojson_data = json.loads(merged_data.to_json())

        with st.container(border=True):
            fig = px.choropleth(
                merged_data, geojson=geojson_data, locations=merged_data.index,
                color=risk_var, hover_name=merged_data.index,
                hover_data={selected_disease: ':,.0f', prevalence_var: ':.2f', risk_var: True},
                color_discrete_map={"Low Risk": "#F6D745", "Medium Risk": "#E55C30", 
                                    "High Risk": "#84206B", "Very High Risk": "#140B34"},
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        
        # INTERPOLATION MAP
        st.subheader('Spatial Interpolation (Trend Surface - Clipped)')
        with st.spinner("Generating interpolation map..."):
            xi, yi, zi = perform_idw_interpolation(merged_data.reset_index(), selected_disease)
            
            if xi is not None:
                fig_interp = go.Figure(data=[go.Contour(
                    z=zi, x=xi[0], y=yi[:, 0], 
                    colorscale='Viridis',
                    connectgaps=False, 
                    contours=dict(start=np.nanmin(zi), end=np.nanmax(zi), size=(np.nanmax(zi) - np.nanmin(zi)) / 20),
                )])
                
                # --- FIX: GANTI NAMA SUMBU ---
                fig_interp.update_layout(
                    title='DBD Risk Interpolation (Clipped)', 
                    height=600,
                    xaxis_title="Longitude",  # Ganti X jadi Longitude
                    yaxis_title="Latitude",   # Ganti Y jadi Latitude
                    yaxis=dict(scaleanchor="x", scaleratio=1) # Biar petanya ga gepeng
                )
                st.plotly_chart(fig_interp, use_container_width=True)
            else:
                st.warning("Interpolation unavailable.")

    # TAB 2: PROVINCE EXPLORER
    with tab2:
        selected_province = st.selectbox("Choose Province:", options=sorted(merged_data.index.unique()))
        province_data = merged_data[merged_data.index == selected_province].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Province Risk Profile")
            risk_level = province_data[risk_var]
            risk_color = get_risk_color(risk_level)
            st.markdown(f"""<div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; color: white;">
                <h2>{selected_province}</h2><h3>{risk_level}</h3></div>""", unsafe_allow_html=True)
            st.write("")
            st.write(f"*Total Cases:* {int(province_data[selected_disease]):,}")
            st.write(f"*Incidence Rate:* {province_data[prevalence_var]:.2f} per 100,000")
        
        with col2:
            st.subheader("Location Map")
            try:
                selected_idx_int = merged_data.index.get_loc(selected_province)
                if isinstance(selected_idx_int, slice): selected_idx_int = selected_idx_int.start
                tetangga_idx = weights.neighbors.get(selected_idx_int, []) if selected_idx_int < len(weights.neighbors) else []
                tetangga_names = [merged_data.index[n] for n in tetangga_idx]
            except: tetangga_names = []
            
            map_data = merged_data.copy()
            map_data['map_category'] = 'Other Provinces'
            if tetangga_names: map_data.loc[map_data.index.isin(tetangga_names), 'map_category'] = 'Neighbor'
            map_data.loc[[selected_province], 'map_category'] = 'Selected Province'
            
            color_map = {'Other Provinces': 'lightgray', 'Selected Province': '#2196F3', 'Neighbor': '#FFC107'}
            fig = px.choropleth(map_data, geojson=geojson_data, locations=map_data.index, color='map_category', color_discrete_map=color_map)
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: RISK PATTERNS
    with tab3:
        st.subheader("Understanding Disease Prevalence Patterns")
        if weights is not None:
            gdf_calc = merged_data.reset_index()
            morans = calculate_morans_i(gdf_calc, selected_disease, weights)
            if morans:
                col1, col2 = st.columns(2)
                with col1: st.metric("Global Moran's I", f"{morans['Statistic']:.4f}")
                with col2: st.metric("Statistical Significance", "YES" if morans['P_value'] < 0.05 else "NO")
                
                lisa_result = calculate_lisa(gdf_calc, selected_disease, weights)
                if lisa_result is not None:
                    merged_lisa = merged_data.copy()
                    lisa_result = lisa_result.set_index('NAME_1')
                    merged_lisa['Cluster'] = lisa_result['Cluster']
                    cluster_colors_map = {'HH (Hotspot)': '#d73027', 'LL (Coldspot)': '#4575b4', 'HL (Outlier)': '#fee090', 'LH (Outlier)': '#91bfdb', 'Not Significant': '#f7f7f7'}
                    fig = px.choropleth(merged_lisa, geojson=geojson_data, locations=merged_lisa.index, color='Cluster', color_discrete_map=cluster_colors_map, title="LISA Cluster Map")
                    fig.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig, use_container_width=True)

    # TAB 4: STATISTICAL MODEL
    with tab4:
        st.subheader("Factors Affecting DBD Prevalence")
        if weights is not None and len(selected_factors) > 0:
            with st.spinner("Fitting spatial models..."):
                gdf_calc = merged_data.reset_index()
                models = fit_spatial_models(gdf_calc, selected_disease, selected_factors, weights)
            if models:
                comparison_data = []
                for model_name, model in models.items():
                    if model is not None:
                        y = model.y; y_pred = model.predy
                        r2_manual = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
                        comparison_data.append({'Model': model_name.upper(), 'AIC': getattr(model, 'aic', np.nan), 'R²': getattr(model, 'r2', r2_manual)})
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    best_idx = comparison_df['AIC'].idxmin()
                    comparison_df['Status'] = ''
                    comparison_df.loc[best_idx, 'Status'] = '⭐ BEST'
                    st.dataframe(comparison_df.style.highlight_min(subset=['AIC'], color='lightgreen'))
                    
                    selected_model_type = st.selectbox("View Coefficients for:", options=list(models.keys()))
                    model = models[selected_model_type]
                    if model is not None:
                        betas = model.betas; se = np.sqrt(np.diag(model.vm))
                        labels = ['Constant'] + selected_factors
                        if selected_model_type == 'lag': labels.append('Rho')
                        elif selected_model_type == 'error': labels.append('Lambda')
                        if len(betas) > len(labels): labels += [f"Var{i}" for i in range(len(betas)-len(labels))]
                        elif len(betas) < len(labels): labels = labels[:len(betas)]
                        coef_df = pd.DataFrame({'Variable': labels, 'Coef': betas.flatten(), 'SE': se.flatten()})
                        coef_df['T-value'] = coef_df['Coef'] / coef_df['SE']
                        coef_df['P-value'] = [2 * (1 - norm.cdf(np.abs(t))) for t in coef_df['T-value']]
                        coef_df['Signif'] = coef_df['P-value'] < 0.05
                        st.dataframe(coef_df.style.apply(lambda x: ['background-color: lightgreen' if x['Signif'] else '' for _ in x], axis=1))
        else: st.warning("Please select at least one risk factor.")

    # TAB 5: EPIDEMIOLOGICAL MEASURES
    with tab5:
        st.subheader("💊 Epidemiological Associations")
        factor_for_epi = st.selectbox("Select Exposure Factor:", selected_factors)
        if factor_for_epi:
            epi_res = calculate_epi_measures(merged_data.copy(), selected_disease, factor_for_epi)
            if epi_res:
                c1, c2 = st.columns(2)
                with c1: st.metric("Odds Ratio (OR)", f"{epi_res['OR']:.2f}")
                with c2: st.metric("Risk Ratio (RR)", f"{epi_res['RR']:.2f}")
                st.write("### Contingency Table")
                st.table(pd.DataFrame(epi_res['table'], columns=['High DBD', 'Low DBD'], index=['Exposed', 'Unexposed']))

    # TAB 6: DATA SUMMARY
    with tab6:
        st.subheader("Raw Data")
        st.dataframe(merged_data)

    # TAB 7: HOW TO USE
    with tab7:
        st.write("### How to Use")
        st.write("Use the sidebar to select risk factors. Navigate tabs to explore maps, stats, and models.")

if __name__ == "__main__":
    main()