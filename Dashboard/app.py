import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from dashboard_prediction.prediction_function import predict_house_price
from app_visualizations import format_currency, remove_outliers, get_numeric_columns, show_data_overview, show_distributions, show_relationships, show_advanced_analysis, show_advanced_analysis_chicago

plt.rcParams['figure.figsize'] = (5, 5) 
# Set page config
st.set_page_config(
    layout="wide", 
    page_title="Real Estate Analytics Pro",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

# Load data functions with caching
@st.cache_data
def load_california():
    try:
        df = pd.read_csv("data/california_data.csv")
        df['city'] = 'California'
        # Create price_per_sqft if not exists
        if 'price_per_sqft' not in df.columns and 'sqft' in df.columns and 'price' in df.columns:
            df['price_per_sqft'] = df['price'] / df['sqft']
        return df
    except FileNotFoundError:
        st.error("California data file not found. Using sample data.")
        return pd.DataFrame({
            'price': np.random.normal(500000, 200000, 1000),
            'sqft': np.random.normal(2000, 500, 1000),
            'bed': np.random.randint(1, 6, 1000),
            'bath': np.random.uniform(1, 4, 1000),
            'zipcode': np.random.randint(90000, 96100, 1000),
            'city': 'California',
            'price_per_sqft': np.random.normal(250, 50, 1000)
        })

@st.cache_data
def load_chicago():
    try:
        df = pd.read_csv("data/chicago_data.csv")
        df['city'] = 'Chicago'
        # Create price_per_sqft if not exists
        if 'price_per_sqft' not in df.columns and 'sqft' in df.columns and 'listPrice' in df.columns:
            df['price_per_sqft'] = df['listPrice'] / df['sqft']
        return df
    except FileNotFoundError:
        st.error("Chicago data file not found. Using sample data.")
        return pd.DataFrame({
            'listPrice': np.random.normal(300000, 150000, 1000),
            'sqft': np.random.normal(1500, 400, 1000),
            'beds': np.random.randint(1, 5, 1000),
            'baths': np.random.uniform(1, 3, 1000),
            'zipcode': np.random.randint(60000, 60800, 1000),
            'city': 'Chicago',
            'price_per_sqft': np.random.normal(200, 40, 1000)
        })

@st.cache_data
def load_redfin():
    try:
        df = pd.read_csv("data/redfin_data.csv")
        df['city'] = 'Redfin'
        # Create price_per_sqft if not exists
        if 'price_per_sqft' not in df.columns and 'Area (SQFT)' in df.columns and 'Price (USD)' in df.columns:
            df['price_per_sqft'] = df['Price (USD)'] / df['Area (SQFT)']
        return df
    except FileNotFoundError:
        st.error("Redfin data file not found. Using sample data.")
        return pd.DataFrame({
            'Price (USD)': np.random.normal(400000, 180000, 1000),
            'Area (SQFT)': np.random.normal(1800, 450, 1000),
            'Beds': np.random.randint(1, 5, 1000),
            'Baths': np.random.uniform(1, 3, 1000),
            'zipcode': np.random.randint(10000, 99900, 1000),
            'city': 'Redfin',
            'price_per_sqft': np.random.normal(220, 45, 1000)
        })

@st.cache_data
def load_merged_data():
    try:
        df = pd.read_csv("data/merged_data.csv")
        return df
    except FileNotFoundError:
        st.error(" data file not found. Using sample data.")
        return pd.DataFrame({
            'Price (USD)': np.random.normal(400000, 180000, 1000),
            'Area (SQFT)': np.random.normal(1800, 450, 1000),
            'Beds': np.random.randint(1, 5, 1000),
            'Baths': np.random.uniform(1, 3, 1000),
            'zipcode': np.random.randint(10000, 99900, 1000),
            'city': 'Redfin',
            'price_per_sqft': np.random.normal(220, 45, 1000)
        })
    

def show_price_prediction_section():
    st.header("🔮 House Price Prediction (Nationwide)")
    st.markdown("""
    Predict house prices using our machine learning model trained on nationwide merged data.
    """)
    
    # Load Redfin data to get available zipcodes
    merged_df = load_merged_data()
    available_zips = sorted(merged_df['zipcode'].astype(str).unique())
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            beds = st.number_input("Number of Bedrooms", 
                                 min_value=1, max_value=10, value=3)
            baths = st.number_input("Number of Bathrooms", 
                                  min_value=1.0, max_value=10.0, value=2.0, step=0.5)
            year_built = st.number_input("Year Built", 
                                       min_value=1800, max_value=2024, value=2000)
        with col2:
            area = st.number_input("Area (SQFT)", 
                                 min_value=500, max_value=10000, value=1500)
            stories = st.number_input("Number of Stories", 
                                   min_value=1, max_value=5, value=2)
            zipcode = st.selectbox("Zip Code", available_zips)
        
        submit_button = st.form_submit_button("Predict Price")
    
    if submit_button:
        try:
            # Make prediction
            prediction = predict_house_price(
                year_built=year_built,
                bed=beds,
                bath=baths,
                Area_SQFT=area,
                zipcode=zipcode,
                stories=stories,
                dataset_name="merged" 
            )
            
            if prediction is not None:
                # Show prediction with nice formatting
                st.success(f"### Predicted Price: ${prediction:,.2f}")
                
                # Show comparison to local market
                zip_data = merged_df[merged_df['zipcode'].astype(str) == zipcode]
                if not zip_data.empty:
                    zip_avg_price = zip_data['price'].mean()
                    price_diff = prediction - zip_avg_price
                    diff_percent = (price_diff / zip_avg_price) * 100
                    
                    st.write("**Market Comparison**")
                    col1, col2 = st.columns(2)
                    col1.metric("Average Price in Zip", f"${zip_avg_price:,.2f}")
                    col2.metric("Difference", 
                               f"${price_diff:,.2f}", 
                               f"{diff_percent:+.1f}%",
                               delta_color="inverse" if price_diff < 0 else "normal")
                    
                    # Show distribution of prices in the zipcode
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.histplot(zip_data['price'], kde=True, ax=ax)
                    ax.axvline(prediction, color='r', linestyle='--', label='Prediction')
                    ax.set_title(f"Price Distribution in Zip {zipcode}")
                    ax.set_xlabel("Price")
                    ax = format_currency(ax, 'x')
                    ax.legend()
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def analyze_california(df, remove_outliers_flag, sample_size, title_props):
    st.header("📊 California Real Estate Market Analysis")
    
    # Data preprocessing
    if remove_outliers_flag:
        numeric_cols = get_numeric_columns(df)
        df = remove_outliers(df, numeric_cols)
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Tabs for different sections - now with 5 tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Relationships", "🧩 Advanced"])
    
    with tab1:
        show_data_overview(df)
    
    with tab2:
        show_distributions(df, title_props)
    
    with tab3:
        show_relationships(df, title_props)
    
    with tab4:
        show_advanced_analysis(df, title_props)

def analyze_chicago(df, remove_outliers_flag, sample_size, title_props):
    st.header("📊 Chicago Real Estate Market Analysis")
    
    # Data preprocessing
    if remove_outliers_flag:
        numeric_cols = get_numeric_columns(df)
        df = remove_outliers(df, numeric_cols)
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Relationships", "🧩 Advanced"])
    
    with tab1:
        show_data_overview(df)
    
    with tab2:
        show_distributions(df, title_props)
    
    with tab3:
        show_relationships(df, title_props)
    
    with tab4:
        show_advanced_analysis_chicago(df, title_props)

def analyze_redfin(df, remove_outliers_flag, sample_size, title_props):
    st.header("📊 Redfin Housing Market Analysis")
    
    # Data preprocessing
    if remove_outliers_flag:
        numeric_cols = get_numeric_columns(df)
        df = remove_outliers(df, numeric_cols)
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Relationships", "🧩 Advanced"])
    
    with tab1:
        show_data_overview(df)
    
    with tab2:
        show_distributions(df, title_props)
    
    with tab3:
        show_relationships(df, title_props)
    
    with tab4:
        show_advanced_analysis(df, title_props)

# Main app
def main():
    st.title("🏠 Real Estate Analytics Pro")
    st.markdown("""
    **A comprehensive dashboard for real estate market analysis across multiple cities.**
    Explore property distributions, correlations, and market trends with interactive visualizations.
    """)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📊 Market Analysis", "🔮 Price Prediction"])
    
    with tab1:
        # Existing city analysis code
        with st.sidebar:
            st.header("Analysis Settings")
            city = st.selectbox("Select City Dataset", ["California", "Chicago", "Redfin"])
            
            st.subheader("Data Options")
            remove_outliers_flag = st.checkbox("Remove Outliers (IQR Method)", True)
            sample_size = st.slider("Sample Size (for large datasets)", 100, 5000, 1000)
            
            st.subheader("Visualization Options")
            theme = st.selectbox("Chart Theme", ["ggplot", "dark_background"])
            plt.style.use(theme)
            title_color = 'black' if theme == 'ggplot' else 'white'
            title_props = {'fontsize': 12, 'fontweight': 'bold', 'color': title_color}
        
        # Load data based on selection
        if city == "California":
            df = load_california()
            analyze_california(df, remove_outliers_flag, sample_size, title_props)
        elif city == "Chicago":
            df = load_chicago()
            analyze_chicago(df, remove_outliers_flag, sample_size, title_props)
        else:
            df = load_redfin()
            analyze_redfin(df, remove_outliers_flag, sample_size, title_props)
    
    with tab2:
        show_price_prediction_section()

if __name__ == "__main__":
    main()