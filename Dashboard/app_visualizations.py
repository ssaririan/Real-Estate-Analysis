import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dashboard_prediction.prediction_function import predict_house_price

def format_currency(ax, axis='y'):
    if axis == 'y':
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'))
    return ax

def remove_outliers(df, columns, threshold=1.5):
    df_clean = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def get_numeric_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def show_data_overview(df):
    st.subheader("Data Overview")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Dataset Summary**")
        st.dataframe(df.describe().style.format("{:.2f}"))
    
    with col2:
        st.write("**Sample Records**")
        st.dataframe(df.head(10))
    
    st.write("**Missing Values Analysis**")
    missing = df.isna().sum().to_frame("Missing Values")
    missing["Percentage"] = (missing["Missing Values"] / len(df)) * 100
    st.dataframe(missing.style.format({"Percentage": "{:.2f}%"}))

def show_distributions(df, title_props):
    st.subheader("Variable Distributions")
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    col1, col2 = st.columns(2)
    with col1:
        var = st.selectbox("Select Variable", numeric_cols, index=0)
    
    with col2:
        plot_type = st.radio("Plot Type", ["Histogram", "Box Plot", "Violin Plot", "ECDF"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define price columns to check against
    price_cols = ['price', 'price_per_sqft', 'zip_mean_price', 'listPrice', 'Price (USD)']
    is_price_var = var in price_cols
    
    if plot_type == "Histogram":
        sns.histplot(df[var], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {var}", **title_props)
        ax.set_ylabel("Count of Properties")
        
        if is_price_var:
            # Format x-axis as currency without scientific notation
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, pos: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'
            ))
            # Only apply ticklabel_format if it's a ScalarFormatter
            if isinstance(ax.xaxis.get_major_formatter(), ticker.ScalarFormatter):
                ax.ticklabel_format(axis='x', style='plain')
            
    elif plot_type == "Box Plot":
        sns.boxplot(x=df[var], ax=ax)
        ax.set_title(f"Box Plot of {var}", **title_props)
        
        # Calculate statistics
        stats = df[var].describe()
        textstr = '\n'.join([
            f'N = {int(stats["count"])}',
            f'Mean = ${stats["mean"]/1e6:.2f}M' if is_price_var else f'Mean = {stats["mean"]:.2f}',
            f'Median = ${stats["50%"]/1e6:.2f}M' if is_price_var else f'Median = {stats["50%"]:.2f}',
            f'Q1 = ${stats["25%"]/1e6:.2f}M' if is_price_var else f'Q1 = {stats["25%"]:.2f}',
            f'Q3 = ${stats["75%"]/1e6:.2f}M' if is_price_var else f'Q3 = {stats["75%"]:.2f}'
        ])
        
        # Get current style (fixed version)
        current_style = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # Checks the first color in cycle
        is_ggplot = current_style == '#E24A33'  # Default ggplot first color
        
        # Theme-aware text box
        text_color = 'black' if is_ggplot else 'white'
        bbox_color = 'white' if is_ggplot else 'black'
        
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                color=text_color,
                bbox=dict(facecolor=bbox_color, alpha=0.7, edgecolor='none'))
        
        if is_price_var:
            ticks = ax.get_xticks()
            ax.set_xticklabels([f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K' for x in ticks])
            
    elif plot_type == "Violin Plot":
        sns.violinplot(x=df[var], ax=ax)
        ax.set_title(f"Violin Plot of {var}", **title_props)
        
        current_style = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # Checks the first color in cycle
        is_ggplot = current_style == '#E24A33'  # Default ggplot first color
        # Theme-aware text box
        text_color = 'black' if is_ggplot else 'white'
        bbox_color = 'white' if is_ggplot else 'black'

        # Calculate statistics
        stats = df[var].describe()
        textstr = '\n'.join([
            f'N = {int(stats["count"])}',
            f'Mean = ${stats["mean"]/1e6:.2f}M' if is_price_var else f'Mean = {stats["mean"]:.2f}',
            f'Median = ${stats["50%"]/1e6:.2f}M' if is_price_var else f'Median = {stats["50%"]:.2f}',
            f'Min = ${stats["min"]/1e6:.2f}M' if is_price_var else f'Min = {stats["min"]:.2f}',
            f'Max = ${stats["max"]/1e6:.2f}M' if is_price_var else f'Max = {stats["max"]:.2f}'
        ])
        
        # Same theme detection as above
        current_style = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        is_ggplot = current_style == '#E24A33'
        
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                color=text_color,
                bbox=dict(facecolor=bbox_color, alpha=0.7, edgecolor='none'))
        
        if is_price_var:
            ticks = ax.get_xticks()
            ax.set_xticklabels([f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K' for x in ticks])
            
    else:  # ECDF
        x = np.sort(df[var].dropna())
        y = np.arange(1, len(x)+1) / len(x)
        ax.plot(x, y, marker='.', linestyle='none')
        ax.set_title(f"ECDF of {var}", **title_props)
        ax.set_ylabel("ECDF")
        
        if is_price_var:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, pos: f'${x/1e6:.1f}M' if x >= 1e6 else f'${x/1e3:.0f}K'
            ))
            if isinstance(ax.xaxis.get_major_formatter(), ticker.ScalarFormatter):
                ax.ticklabel_format(axis='x', style='plain')
    
    # Set figure background color to white
    #fig.patch.set_facecolor('white')
    st.pyplot(fig)
    

def show_relationships(df, title_props):
    st.subheader("Feature Relationships")
    
    numeric_cols = get_numeric_columns(df)
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X Variable", numeric_cols, 
                           index=numeric_cols.index('sqft') if 'sqft' in numeric_cols else 
                           numeric_cols.index('Area (SQFT)') if 'Area (SQFT)' in numeric_cols else 0)
    
    with col2:
        y_var = st.selectbox("Y Variable", numeric_cols, 
                           index=numeric_cols.index('price') if 'price' in numeric_cols else 
                           numeric_cols.index('listPrice') if 'listPrice' in numeric_cols else
                           numeric_cols.index('Price (USD)') if 'Price (USD)' in numeric_cols else 1)
    
    plot_type = st.radio("Visualization Type", ["Scatter Plot", "Hexbin Plot", "Regression Plot", "2D Density"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if plot_type == "Scatter Plot":
        sns.scatterplot(x=df[x_var], y=df[y_var], alpha=0.6, ax=ax)
    elif plot_type == "Hexbin Plot":
        ax.hexbin(df[x_var], df[y_var], gridsize=30, cmap='Blues')
        cb = plt.colorbar(ax.collections[0])
        cb.set_label('Counts')
    elif plot_type == "Regression Plot":
        sns.regplot(x=df[x_var], y=df[y_var], ax=ax, scatter_kws={'alpha':0.3})
    else:  # 2D Density
        sns.kdeplot(x=df[x_var], y=df[y_var], fill=True, cmap='Blues', ax=ax)
    
    plt.title(f"{y_var} vs {x_var}", **title_props)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    if y_var in ['price', 'price_per_sqft', 'zip_mean_price', 'listPrice', 'Price (USD)']:
        ax = format_currency(ax, 'y')
    if x_var in ['price', 'price_per_sqft', 'zip_mean_price', 'listPrice', 'Price (USD)']:
        ax = format_currency(ax, 'x')
    
    st.pyplot(fig)
    
    # Correlation stats
    st.write("**Correlation Statistics**")
    corr = df[[x_var, y_var]].corr().iloc[0,1]
    st.metric("Pearson Correlation Coefficient", f"{corr:.3f}")

def show_advanced_analysis(df, title_props):
    st.subheader("Advanced Analytics")
    
    option = st.selectbox("Select Analysis", 
                         ["Price Distribution by Zip Code", 
                          "Price per Sqft Analysis",
                          "Market Segmentation (PCA)"])
    
    if option == "Price Distribution by Zip Code":
        top_n = st.slider("Number of Zip Codes to Display", 5, 20, 10)
        
        # Determine price column name based on dataset
        price_col = 'price' if 'price' in df.columns else 'listPrice' if 'listPrice' in df.columns else 'Price (USD)'
        price_per_sqft_col = 'price_per_sqft'
        
        metric = st.radio("Rank Zip Codes By", ["Mean Price", "Median Price", "Price per Sqft"])
        
        if metric == "Mean Price":
            top_zips = df.groupby('zipcode')[price_col].mean().nlargest(top_n).index
            title = f"Top {top_n} Zip Codes by Mean Price"
            y_var = price_col
        elif metric == "Median Price":
            top_zips = df.groupby('zipcode')[price_col].median().nlargest(top_n).index
            title = f"Top {top_n} Zip Codes by Median Price"
            y_var = price_col
        else:
            top_zips = df.groupby('zipcode')[price_per_sqft_col].mean().nlargest(top_n).index
            title = f"Top {top_n} Zip Codes by Price per Sqft"
            y_var = price_per_sqft_col
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=df[df['zipcode'].isin(top_zips)]['zipcode'].astype(str), 
                    y=df[y_var], ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(title, **title_props)  # Applied title_props here
        plt.xlabel("Zip Code")
        plt.ylabel(y_var.replace('_', ' ').title())
        
        if y_var in [price_col, price_per_sqft_col]:
            ax = format_currency(ax, 'y')
        
        st.pyplot(fig)
        
        # Show top zip codes table
        st.write(f"**{title}**")
        if metric == "Mean Price":
            st.dataframe(df.groupby('zipcode')[price_col].mean().nlargest(top_n).to_frame("Mean Price").style.format("${:,.2f}"))
        elif metric == "Median Price":
            st.dataframe(df.groupby('zipcode')[price_col].median().nlargest(top_n).to_frame("Median Price").style.format("${:,.2f}"))
        else:
            st.dataframe(df.groupby('zipcode')[price_per_sqft_col].mean().nlargest(top_n).to_frame("Mean Price per Sqft").style.format("${:,.2f}"))

    elif option == "Price per Sqft Analysis":
        st.write("**Price per Square Foot Distribution**")

        categorical_cols = get_categorical_columns(df)
        hue_var = st.selectbox("Color By", ['None'] + categorical_cols)

        price_per_sqft_col = 'price_per_sqft'
        sqft_col = 'sqft' if 'sqft' in df.columns else 'Area (SQFT)'
        price_col = (
        'price' if 'price' in df.columns else
        'listPrice' if 'listPrice' in df.columns else
        'Price (USD)'
        )
        bed_col = (
        'bed' if 'bed' in df.columns else
        'beds' if 'beds' in df.columns else
        'Beds'
        )
        bath_col = (
        'bath' if 'bath' in df.columns else
        'baths' if 'baths' in df.columns else
        'Baths'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        if hue_var == 'None':
            sns.histplot(df[price_per_sqft_col], kde=True, bins=30, ax=ax)
        else:
            sns.histplot(
            df, x=price_per_sqft_col, hue=hue_var,
            element='step', kde=True, ax=ax
            )

        ax.set_title("Price per Square Foot Distribution", **title_props)

        # Format x-axis to show regular dollar amounts
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f'${x:.0f}' if x == int(x) else f'${x:.2f}'
        ))

        plt.xlabel("Price per Sqft ($)")
        st.pyplot(fig)

        # Interactive plot with plotly
        st.write("**Interactive Price vs. Sqft**")
        fig2 = px.scatter(
            df, x=sqft_col, y=price_col,
            color=hue_var if hue_var != 'None' else None,
            hover_data=(
            ['zipcode', bed_col, bath_col] if 'zipcode' in df.columns
            else [bed_col, bath_col]
            ),
            title="Price vs. Square Footage"
        )
        fig2.update_yaxes(tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)


    
    else:  # Market Segmentation (PCA)
        st.write("**Market Segmentation with PCA**")
        
        numeric_cols = get_numeric_columns(df)
        default_features = []
        for col in ['price', 'sqft', 'bed', 'bath', 'price_per_sqft', 'listPrice', 'Area (SQFT)', 'Beds', 'Baths']:
            if col in numeric_cols and col not in default_features:
                default_features.append(col)
                if len(default_features) >= 5:
                    break
        
        selected_features = st.multiselect("Select Features for PCA", 
                                         numeric_cols,
                                         default=default_features[:5])
        
        if len(selected_features) >= 2:
            # Standardize data
            X = df[selected_features].dropna()
            X_std = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_std)
            pca_df = pd.DataFrame(data=principal_components, 
                                 columns=['PC1', 'PC2'])
            
            # Add hue variable if selected
            categorical_cols = get_categorical_columns(df)
            hue_var = st.selectbox("Color Points By", ['None'] + categorical_cols)
            if hue_var != 'None':
                pca_df[hue_var] = df.loc[X.index, hue_var].values
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if hue_var == 'None':
                sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6, ax=ax)
            else:
                sns.scatterplot(x='PC1', y='PC2', hue=hue_var, data=pca_df, alpha=0.6, ax=ax)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            ax.set_title("PCA of Real Estate Features", **title_props)  # Applied title_props here
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            st.pyplot(fig)
            
            # Show component loadings
            st.write("**PCA Component Loadings**")
            loadings = pd.DataFrame(pca.components_.T, 
                                  columns=['PC1', 'PC2'],
                                  index=selected_features)
            st.dataframe(loadings.style.format("{:.2f}").background_gradient(cmap='Blues'))
        else:
            st.warning("Please select at least 2 features for PCA analysis.")

def show_advanced_analysis_chicago(df, title_props):
    st.subheader("Advanced Analytics")
    
    option = st.selectbox("Select Analysis", 
                         ["Price per Sqft Analysis",
                          "Market Segmentation (PCA)"])
    
    if option == "Price per Sqft Analysis":
        st.write("**Price per Square Foot Distribution**")

        categorical_cols = get_categorical_columns(df)
        hue_var = st.selectbox("Color By", ['None'] + categorical_cols)

        price_per_sqft_col = 'price_per_sqft'
        sqft_col = 'sqft' if 'sqft' in df.columns else 'Area (SQFT)'
        price_col = (
        'price' if 'price' in df.columns else
        'listPrice' if 'listPrice' in df.columns else
        'Price (USD)'
        )
        bed_col = (
        'bed' if 'bed' in df.columns else
        'beds' if 'beds' in df.columns else
        'Beds'
        )
        bath_col = (
        'bath' if 'bath' in df.columns else
        'baths' if 'baths' in df.columns else
        'Baths'
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        if hue_var == 'None':
            sns.histplot(df[price_per_sqft_col], kde=True, bins=30, ax=ax)
        else:
            sns.histplot(
            df, x=price_per_sqft_col, hue=hue_var,
            element='step', kde=True, ax=ax
            )

        ax.set_title("Price per Square Foot Distribution", **title_props)

        # Format x-axis to show regular dollar amounts
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f'${x:.0f}' if x == int(x) else f'${x:.2f}'
        ))

        plt.xlabel("Price per Sqft ($)")
        st.pyplot(fig)

        # Interactive plot with plotly
        st.write("**Interactive Price vs. Sqft**")
        fig2 = px.scatter(
            df, x=sqft_col, y=price_col,
            color=hue_var if hue_var != 'None' else None,
            hover_data=(
            ['zipcode', bed_col, bath_col] if 'zipcode' in df.columns
            else [bed_col, bath_col]
            ),
            title="Price vs. Square Footage"
        )
        fig2.update_yaxes(tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)
    
    else:  # Market Segmentation (PCA)
        st.write("**Market Segmentation with PCA**")
        
        numeric_cols = get_numeric_columns(df)
        default_features = []
        for col in ['price', 'sqft', 'bed', 'bath', 'price_per_sqft', 'listPrice', 'Area (SQFT)', 'Beds', 'Baths']:
            if col in numeric_cols and col not in default_features:
                default_features.append(col)
                if len(default_features) >= 5:
                    break
        
        selected_features = st.multiselect("Select Features for PCA", 
                                         numeric_cols,
                                         default=default_features[:5])
        
        if len(selected_features) >= 2:
            # Standardize data
            X = df[selected_features].dropna()
            X_std = StandardScaler().fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_std)
            pca_df = pd.DataFrame(data=principal_components, 
                                 columns=['PC1', 'PC2'])
            
            # Add hue variable if selected
            categorical_cols = get_categorical_columns(df)
            hue_var = st.selectbox("Color Points By", ['None'] + categorical_cols)
            if hue_var != 'None':
                pca_df[hue_var] = df.loc[X.index, hue_var].values
            
            # Plot
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color=hue_var if hue_var != 'None' else None,
                title="PCA of Real Estate Features",
                labels={
                'PC1': f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                'PC2': f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
                },
                opacity=0.7
                )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show component loadings
            st.write("**PCA Component Loadings**")
            loadings = pd.DataFrame(pca.components_.T, 
                                  columns=['PC1', 'PC2'],
                                  index=selected_features)
            st.dataframe(loadings.style.format("{:.2f}").background_gradient(cmap='Blues'))
        else:
            st.warning("Please select at least 2 features for PCA analysis.")
