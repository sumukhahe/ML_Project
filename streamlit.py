import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", page_title="Agricultural Dashboard")
# Load the datasets
@st.cache_data
def load_data():
    crop_data = pd.read_csv('crop_data.csv')
    area_affected = pd.read_csv('Area_affected.csv')
    mgnrega = pd.read_csv('mgnrega.csv')
    return crop_data, area_affected, mgnrega

crop_data, area_affected, mgnrega = load_data()



# Custom CSS for styling
st.markdown("""
<style>
    .stSelectbox {margin-bottom: 0px;}
    .stTab {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stTab[data-baseweb="tab"] {
        height: 50px;
        white-space: normal;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Breadcrumb
st.title("Rural Distress and Mgnrega")

# State and view dropdowns
col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State:", [""] + sorted(mgnrega['State'].unique()))
with col2:
    view = st.selectbox("View:", ["", "Data", "Visualization"])

# Only display content if both dropdowns have been selected
if state and view:
    if view == "Data":
        st.subheader(f"Data for {state}")
        st.info(""" * **State:** The geographic region or state where the MGNREGA data is reported.\n * **Rural_Population:** The total population living in rural areas within the state.\n * **year:** The year in which the data was recorded.\n * **No_of_Registered:** The number of individuals registered for MGNREGA work.\n * **Employment_demanded:** The total number of employment days demanded by registered individuals.\n * **Employment_offered:** The total number of employment days offered to individuals.\n * **Employment_Availed:** The total number of employment days availed by individuals.\n """)
        # Display MGNREGA data
        st.write("MGNREGA Data:")
        st.dataframe(mgnrega[mgnrega['State'] == state])
        
        st.info(""" * **Crop:** Type of crop being reported.\n * **State:** Geographic region or state where the crop is grown.\n * **Crop_Year:** The year in which the crop was grown or harvested.\n * **Area_(in_Ha):** Total area (in hectares) of land used for growing the crop.\n * **Production_(in_Tonnes):** Total amount of crop produced, measured in tonnes.\n * **Yield_(kg/Ha):** Average yield of the crop per hectare, measured in kilograms.\n * **MSP:** Minimum Support Price, the price at which the government guarantees to buy the crop.\n * **Annual_rainfall:** Total amount of rainfall received in a year, affecting crop growth.\n * **Cost_of_prod:** Cost incurred in the production of the crop.\n * **Harvest_Price:** Selling price of the crop at harvest time.""")
        # Display Crop data
        st.write("Crop Data:")
        st.dataframe(crop_data[crop_data['State'] == state])
        
        st.info(""" * **Year:** The year in which the data was recorded.\n * **State:** The geographic region or state where the crop area damage is reported.\n * **Total Area of State:** The total crop area of the state.\n * **Area_aff:** The area affected by crop-related issues or factors.\n * **Wages:** The wages paid, likely related to agricultural work or compensation in the affected area.""")
        # Display Area Affected data
        st.write("Area Affected Data:")
        st.dataframe(area_affected[area_affected['State'] == state])
    
    elif view == "Visualization":
        # Filter data for the selected state
        state_mgnrega = mgnrega[mgnrega['State'] == state].sort_values('year')
        state_crop = crop_data[crop_data['State'] == state].sort_values('Crop_Year')
        state_area = area_affected[area_affected['State'] == state].sort_values('Year')

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        latest_year = 2023
        prev_year = 2022
        
        latest_data = state_mgnrega[state_mgnrega['year'] == latest_year].iloc[0]
        prev_data = state_mgnrega[state_mgnrega['year'] == prev_year].iloc[0] if prev_year in state_mgnrega['year'].values else None

        def calculate_change(current, previous):
            if previous and previous != 0:
                change = (current - previous) / previous * 100
                return f"{change:+.2f}% from previous year"
            return None

        with col1:
            st.metric("No. of Registered", f"{latest_data['No_of_Registered']:,}", 
                      calculate_change(latest_data['No_of_Registered'], prev_data['No_of_Registered']) if prev_data is not None else None)
        with col2:
            st.metric("Employment Demanded", f"{latest_data['Employment_demanded']:,}", 
                      calculate_change(latest_data['Employment_demanded'], prev_data['Employment_demanded']) if prev_data is not None else None)
        with col3:
            st.metric("Employment Offered", f"{latest_data['Employment_offered']:,}", 
                      calculate_change(latest_data['Employment_offered'], prev_data['Employment_offered']) if prev_data is not None else None)
        with col4:
            st.metric("Employment Availed", f"{latest_data['Employment_Availed']:,}", 
                      calculate_change(latest_data['Employment_Availed'], prev_data['Employment_Availed']) if prev_data is not None else None)

       
       # Tabs setup
        tabs = st.tabs(["Summary Statistics", "APY Trends", "Harvest", "Mgnrega", "Conclusion"])

        # Tab 1: APY TRENDS
        with tabs[1]:
            col1, col2 = st.columns(2)

            # First Column: MGNREGA Trends
            
            with col1:
                crop = st.selectbox("Select Crop:", [""] + sorted(crop_data['Crop'].unique()))
        
        # Filter data based on selected crop and state (you've already filtered by state elsewhere)
                if crop:
                    state_crop_data = crop_data[(crop_data['State'] == state) & (crop_data['Crop'] == crop)]
                    
                    st.subheader("APY Trends")
                    st.caption("Area,Production of Crops")
                    
                    if not state_crop_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=state_crop_data['Crop_Year'], y=state_crop_data['Production_(in_Tonnes)'], mode='lines+markers', name='Production', line=dict(color='blue'),hovertemplate="<b>Year</b>: %{x}<br><b>Production</b>: %{y:,}<extra></extra>"))
                        fig.add_trace(go.Scatter(x=state_crop_data['Crop_Year'], y=state_crop_data['Area_(in_Ha)'], mode='lines+markers', name='Area', line=dict(color='green'), yaxis='y2',hovertemplate="<b>Year</b>: %{x}<br><b>Area</b>: %{y:,}<extra></extra>"))

                        fig.update_layout(
                            xaxis_title='Year',
                            yaxis_title='Production',
                            yaxis2=dict(title='Area', overlaying='y', side='right'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data available for the selected crop.")
                else:
                    st.info("Please select a crop to view the trends.")
    # Second Column: Crop Production and Yield
            with col2:
                st.write("<br>", unsafe_allow_html=True)
                st.write("<br>", unsafe_allow_html=True)
                st.write("<br>", unsafe_allow_html=True)
               
                if crop:
                    state_crop_data = crop_data[(crop_data['State'] == state) & (crop_data['Crop'] == crop)]

                    st.subheader("Crop Production and Yield")
                    st.caption("Production,Yield of Crops")
                    if not state_crop_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=state_crop_data['Crop_Year'], y=state_crop_data['Production_(in_Tonnes)'], name='Production',marker_color='#98FB98',hovertemplate="<b>Year</b>: %{x}<br><b>Production</b>: %{y:,}<extra></extra>"))
                        fig.add_trace(go.Scatter(x=state_crop_data['Crop_Year'], y=state_crop_data['Yield_(kg/Ha)'], mode='lines+markers', name='Yield', yaxis='y2',line=dict(color='rgb(0,100,0)'),hovertemplate="<b>Year</b>: %{x}<br><b>Yield</b>: %{y:,}<extra></extra>"))

                        fig.update_layout(
                            title=f"Crop Production and Yield for {state}",
                            xaxis_title="Year",
                            yaxis_title="Production (Tonnes)",
                            yaxis2=dict(title="Yield (kg/Ha)", overlaying='y', side='right'),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

#------------------------------- Mean,Median,std ---------------------------------------------------------

        def display_histograms(data, dataset_name, col1, col2):
            numerical_columns = data.select_dtypes(include=['number']).columns

    # Split columns for two-column display
            mid = len(numerical_columns) // 2
            col_list_1 = numerical_columns[:mid]
            col_list_2 = numerical_columns[mid:]
          

            with col1:
                for column in col_list_1:
                    if not data[column].isnull().all():
                        mean = data[column].mean()
                        median = data[column].median()
                        std_dev = data[column].std()

                        st.subheader(f"{column}")
                        fig, ax = plt.subplots(figsize=(8,4))
                        
                        # Plot histogram
                        sns.histplot(data[column].dropna(), bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)

                        # Generate values for Gaussian curve
                        x = np.linspace(data[column].min(), data[column].max(), 100)
                        gaussian_curve = norm.pdf(x, mean, std_dev) * len(data[column].dropna()) * (data[column].max() - data[column].min()) / 20  # Adjust scaling for bins
                        
                        # Plot Gaussian curve (Mean and Std Dev)
                        ax.plot(x, gaussian_curve, color='red', linestyle='-', label=f"Gaussian Curve (Mean: {mean:.2f}, Std Dev: {std_dev:.2f})", linewidth=2)

                        # Plot Median as a vertical line
                        ax.axvline(median, color='green', linestyle='-', linewidth=2, label=f"Median: {median:.2f}")
                        
                        # Add legend
                        ax.legend()

                        # Display the plot in Streamlit
                        st.pyplot(fig)

            with col2:
                for column in col_list_2:
                    if not data[column].isnull().all():
                        mean = data[column].mean()
                        median = data[column].median()
                        std_dev = data[column].std()

                        st.subheader(f"{column}")
                        fig, ax = plt.subplots(figsize=(8,4))

                        # Plot histogram
                        sns.histplot(data[column].dropna(), bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)

                        # Generate values for Gaussian curve
                        x = np.linspace(data[column].min(), data[column].max(), 100)
                        gaussian_curve = norm.pdf(x, mean, std_dev) * len(data[column].dropna()) * (data[column].max() - data[column].min()) / 20  # Adjust scaling for bins
                        
                        # Plot Gaussian curve (Mean and Std Dev)
                        ax.plot(x, gaussian_curve, color='red', linestyle='-', label=f"Gaussian Curve (Mean: {mean:.2f}, Std Dev: {std_dev:.2f})", linewidth=2)

                        # Plot Median as a vertical line
                        ax.axvline(median, color='green', linestyle='-', linewidth=2, label=f"Median: {median:.2f}")
                        
                        # Add legend
                        ax.legend()

                        # Display the plot in Streamlit
                        st.pyplot(fig)

# --------------- OUTLIERS -------------------------------------------

        # Function to create box plots for outliers
        def create_box_plots(data, dataset_name, col1, col2):
            numerical_columns = data.select_dtypes(include=['number']).columns

    # Split the columns for two-column display
            mid = len(numerical_columns) // 2
            col_list_1 = numerical_columns[:mid]
            col_list_2 = numerical_columns[mid:]

            with col1:
                for column in col_list_1:
                    if not data[column].isnull().all():  # Check if the column has valid data
                        st.subheader(f"{column} Outliers - {dataset_name}")
                        fig = px.box(data, y=column, title=f"Outliers in {column}")
                        st.plotly_chart(fig, use_container_width=True)

            with col2:
                for column in col_list_2:
                    if not data[column].isnull().all():  # Check if the column has valid data
                        st.subheader(f"{column} Outliers - {dataset_name}")
                        fig = px.box(data, y=column, title=f"Outliers in {column}")
                        st.plotly_chart(fig, use_container_width=True)

# ---------------- QQ PLOT ------------------------------------
        def create_qq_plots(data, dataset_name, col1, col2):
            numerical_columns = data.select_dtypes(include=['number']).columns

            # Split columns for two-column display
            mid = len(numerical_columns) // 2
            col_list_1 = numerical_columns[:mid]
            col_list_2 = numerical_columns[mid:]

            with col1:
                for column in col_list_1:
                    if not data[column].isnull().all():  # Check if the column has valid data
                        st.subheader(f"{column} Normality Check (QQ Plot) - {dataset_name}")
                        fig, ax = plt.subplots(figsize=(6,3))
                        sm.qqplot(data[column], line='s', ax=ax)
                        st.pyplot(fig)

            with col2:
                for column in col_list_2:
                    if not data[column].isnull().all():  # Check if the column has valid data
                        st.subheader(f"{column} Normality Check (QQ Plot) - {dataset_name}")
                        fig, ax = plt.subplots(figsize=(6,3))
                        sm.qqplot(data[column], line='s', ax=ax)
                        st.pyplot(fig)


        def create_correlation_plot(data, selected_columns, col1, col2, method='spearman'):
                                if len(selected_columns) >= 2:
                                    data = data[selected_columns]
                                    corr_matrix = data.corr(method=method)
                            
                            # Plot the correlation matrix
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(corr_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, ax=ax, fmt='.2f', cbar=True)
                                    plt.title(f"Spearman Correlation Matrix ({method.capitalize()})", pad=20)
                                    with col1:
                                        st.pyplot(fig)
                                else:
                                    st.info("Please select at least two columns to display correlation.")

# Tab 2: PRICE ANALYSIS INSIGHTS
        with tabs[0]:
            st.subheader("Summary Statistics • Outliers • Normal Distribution • Correlation Analysis")
            analysis_type = st.selectbox("Select Analysis", ["Mean, Median, Std Dev", "Outliers", "QQ Plot","Correlation Analysis" ])

            if analysis_type != "Selwect an Option":
                col1, col2 = st.columns(2)

            # Display the selected type of analysis
                if analysis_type == "Mean, Median, Std Dev":
                    display_histograms(crop_data, "Crop Data", col1, col2)
                    display_histograms(mgnrega, "Production Data", col1, col2)
                    display_histograms(area_affected, "Area Data", col1, col2)

                elif analysis_type == "Outliers":
                    create_box_plots(crop_data, "Crop Data", col1, col2)
                    create_box_plots(mgnrega, "Production Data", col1, col2)
                    create_box_plots(area_affected, "Area Data", col1, col2)

                elif analysis_type == "QQ Plot":
                    create_qq_plots(crop_data, "Crop Data", col1, col2)
                    create_qq_plots(mgnrega, "Production Data", col1, col2)
                    create_qq_plots(area_affected, "Area Data", col1, col2)
                elif analysis_type == "Correlation Analysis":
                    all_datasets = {
                        "Crop Data": crop_data,
                        "Dataset 2": mgnrega,
                        "Dataset 3": area_affected
                        }

            # Combine all datasets into one DataFrame
                    combined_data = pd.concat(all_datasets.values(), axis=1)

            # Dropdown to select columns for correlation
                    numerical_columns = combined_data.select_dtypes(include=['number']).columns
                    selected_columns = st.multiselect("Select Columns to Correlate:", numerical_columns)

                    col1, col2 = st.columns(2)

            # Create correlation plot
                    create_correlation_plot(combined_data, selected_columns, col1, col2)
            else:
                st.info("Please select an analysis type from the dropdown.")

#----------------------- HARVEST PRICE --------------------------------------------------------

        with tabs[2]:
            col1, col2 = st.columns(2)

    # First Column: MGNREGA Trends
            with col2:
                st.subheader("Price Trends")

                if crop and state:
                    state_crop_data = crop_data[(crop_data['State'] == state) & (crop_data['Crop'] == crop)]

                    st.subheader("")
                    st.caption("Cost Production vs Harvest Price")
                    

                    if not state_crop_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=state_crop_data['Crop_Year'], y=state_crop_data['cost_of_prod'], mode='lines+markers', name='Production_Cost', line=dict(color='blue'), hovertemplate="<b>Year</b>: %{x}<br><b>Production Cost</b>: %{y:,}<extra></extra>"))
                        fig.add_trace(go.Scatter(x=state_crop_data['Crop_Year'], y=state_crop_data['Harvest_Price'], mode='lines+markers', name='Harvest Price', line=dict(color='green'), yaxis='y2', hovertemplate="<b>Year</b>: %{x}<br><b>Harvest Price</b>: %{y:,}<extra></extra>"))

                        fig.update_layout(
                            xaxis_title='Year',
                            yaxis_title='Production',
                            yaxis2=dict(title='Area', overlaying='y', side='right'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data available for the selected crop.")
                else:
                    st.info("Please select a crop and state to view the trends.")

            # Second Column: Feature Importance
            with col1:
                st.subheader("Cost Production and Harvest Price")
                st.caption("Feature Importance")

                # Identify categorical columns and the 'Crop_Year' column
                categorical_cols = crop_data.select_dtypes(include=['object']).columns
                year_col = 'Crop_Year'

                # Create a list of columns to scale (exclude categorical and Crop_Year columns)
                cols_to_scale = [col for col in crop_data.columns if col not in categorical_cols and col != year_col and col != 'Harvest_Price']

                # Apply Min-Max scaling
                scaler = MinMaxScaler()
                crop_data[cols_to_scale] = scaler.fit_transform(crop_data[cols_to_scale])

                # Initialize a dictionary to store feature importances for each year
                feature_importance_dict = {}

                # Separate the dataset by Crop_Year
                grouped = crop_data.groupby('Crop_Year')

                for year, group in grouped:
                    # Separate features (X) and target (y)
                    X = group[cols_to_scale]  # All columns except 'Crop_Year', categorical, and 'Harvest_Price'
                    y = group['Harvest_Price']  # Use 'Harvest_Price' as the target variable
                    
                    # Train Random Forest model
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    
                    # Get feature importance for the current year
                    feature_importance = rf.feature_importances_
                    feature_importance_dict[year] = feature_importance

                # Plot feature importance for each year
                plt.figure(figsize=(14, 8))
                feature_names = cols_to_scale
                for i, year in enumerate(sorted(feature_importance_dict.keys())):
                    plt.bar([f"{feature}\n{year}" for feature in feature_names], feature_importance_dict[year], alpha=0.7, label=f"Year {year}")

               
                plt.xlabel('Features and Year', fontsize=14)
                plt.ylabel('Feature Importance', fontsize=14)
                plt.xticks(rotation=90, fontsize=10)
                plt.yticks(fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend(title='Year')

                st.pyplot(plt.gcf())
                plt.close()
            

#-------------------------------------- EMPLOYMENT DEMANDED---------------------------

        def scale_data(df):
            target_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered', 'Employment_Availed']
            
            # Apply Min-Max scaling
            scaler = MinMaxScaler()
            df[target_cols] = scaler.fit_transform(df[target_cols])
            
            return df

       
        scaled_mgnrega = scale_data(mgnrega)

        # Streamlit app

        with tabs[3]:
            col1, col2 = st.columns(2)

            # First Column: Employment Trends
            with col1:
                st.subheader("Employment Trends")
                st.caption("Employment Demanded vs Employment Offered")
                year = st.selectbox("Select Year:", sorted(scaled_mgnrega['year'].unique()))

                if state:
                    state_data = scaled_mgnrega[scaled_mgnrega['State'] == state]

                    if not state_data.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=state_data['year'], 
                            y=state_data['Employment_demanded'], 
                            mode='lines+markers', 
                            name='Employment Demanded', 
                            line=dict(color='blue'), 
                            hovertemplate="<b>Year</b>: %{x}<br><b>Employment Demanded</b>: %{y:,}<extra></extra>"
                        ))
                        fig.add_trace(go.Scatter(
                            x=state_data['year'], 
                            y=state_data['Employment_offered'], 
                            mode='lines+markers', 
                            name='Employment Offered', 
                            line=dict(color='green'), 
                            hovertemplate="<b>Year</b>: %{x}<br><b>Employment Offered</b>: %{y:,}<extra></extra>"
                        ))

                        fig.update_layout(
                            xaxis_title='Year',
                            yaxis_title='Employment Demanded',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data available for the selected state.")
                else:
                    st.info("Please select a state to view the trends.")

            # Second Column: Feature Importance as Pie Chart
            with col2:
                st.subheader("Feature Importance")
                
                if year:
                    year_df = scaled_mgnrega[scaled_mgnrega['year'] == year]
                    feature_cols = ['Rural_Population', 'No_of_Registered', 'Employment_demanded', 'Employment_offered']
                    target_col = 'Employment_Availed'
                    
                    X = year_df[feature_cols]
                    y = year_df[target_col]
                    
                    model = RandomForestRegressor()
                    model.fit(X, y)
                    
                    importances = model.feature_importances_
                    features = X.columns
                    colors = [ '#eb5f1a','#f6a417', '#66c6de', '#fecf16']

# Ensure the color palette length matches the number of features
                    color_palette = colors[:len(features)]  # This trims the color palette if there are fewer features than colors

                    # Plot feature importance as a pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(importances, labels=features, autopct='%1.1f%%', startangle=90, colors=color_palette)
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.info("Please select a year to view the feature importance.")

        with tabs[4]:
            st.info("* **Feature Importance Distribution :** Feature Importance is done using Random forest Regressor ,Features consistently ranked highly over multiple years strongly influence the target variable (msp , production ,yield). Variability in importance suggests changes in external factors .")
            st.info("* **Year-wise Comparison :** A consistent feature importance across multiple years suggests that the relationship between input features and the target variable remains relatively stable, indicating the model is effectively capturing long-term trends. On the other hand, significant year-over-year changes in feature importance may imply the model needs to adjust to new patterns, such as evolving agricultural practices or changing market conditions.")
            st.info("* **Impact on Decision-Making :** If features like Employment Availed or Employment Offered grow in importance, it signals a stronger influence of labor factors on outcomes, guiding decision-makers to focus on related policies. Conversely, decreasing importance of certain features suggests they may be losing relevance, warranting a review of their role in decision-making and modeling.")
else:
    st.info("Please select a State and View option to display the dashboard.")
