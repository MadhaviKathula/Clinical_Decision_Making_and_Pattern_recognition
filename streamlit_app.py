import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt


# Function to load data
@st.cache_data
def load_data():
    data = pd.read_csv('healthcare_dataset.csv')
    # Convert date columns to datetime
    data['Date of Admission'] = pd.to_datetime(data['Date of Admission'], errors='coerce')
    data['Discharge Date'] = pd.to_datetime(data['Discharge Date'], errors='coerce')
    # Calculate Length of Stay
    data['Length of Stay'] = (data['Discharge Date'] - data['Date of Admission']).dt.days
    return data


# Load the data
df = load_data()

# Streamlit app title
st.title('Healthcare Data Insights')

# Sidebar for filters
st.sidebar.title('Filters')
selected_gender = st.sidebar.selectbox('Select Gender', options=['All'] + list(df['Gender'].unique()))
selected_condition = st.sidebar.selectbox('Select Medical Condition',
                                          options=['All'] + list(df['Medical Condition'].unique()))
#selected_hospital = st.sidebar.selectbox('Select Hospital', options=['All'] + list(df['Hospital'].unique()))

# Filter data based on the selected filters
filtered_df = df.copy()
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
if selected_condition != 'All':
    filtered_df = filtered_df[filtered_df['Medical Condition'] == selected_condition]
#if selected_hospital != 'All':
#    filtered_df = filtered_df[filtered_df['Hospital'] == selected_hospital]

# Data overview
st.sidebar.title('Data Overview')
st.sidebar.write(f"Total entries: {len(filtered_df)}")
st.sidebar.write(f"Average Length of Stay: {filtered_df['Length of Stay'].mean():.2f} days")
st.sidebar.write(f"Average Billing Amount: {filtered_df['Billing Amount'].mean():.2f}")

# Create tabs for different insights
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Patient Demographics", "Financial Insights", "Medical Condition Analysis", "Correlation Analysis"])

with tab0:
    st.header("Overview")
    st.markdown("""
    
    - **Chain Reasoning & Agentic Generative AI:** The app provides a foundation for data-driven decision-making processes by filtering and visualizing patient data, allowing for deeper insights and more informed decisions.
    
    - **Classification, Prediction & Inference:** The various visualizations, such as gender distribution and age distribution, support the identification and classification of patterns within the patient data. This can be crucial for predictive analytics and inferring potential health outcomes.
    
    - **Clustering & Time-Series Anomaly Detection:** The analysis of Length of Stay over time and correlation heatmaps offer insights into clustering behaviors and help in detecting anomalies in the data, which is vital for optimizing treatment, payment, and operations (TPO).
    
    This application not only serves as a tool for exploratory data analysis but also aligns closely with modern healthcare analytics frameworks, facilitating improved clinical decision-making and pattern recognition.
    """)

with tab1:
    st.header("Patient Demographics Overview")

    # Gender Distribution by Medical Condition
    st.subheader("Gender Distribution by Medical Condition")
    gender_condition = filtered_df.groupby(['Medical Condition', 'Gender']).size().unstack().fillna(0)
    custom_colors = ['green', 'red']
    fig = px.bar(gender_condition, title='Gender Distribution by Medical Condition',
                 labels={'value': 'Number of Patients'},
                 color_discrete_sequence=custom_colors)
    fig.update_traces(hovertemplate='%{y} patients')
    st.plotly_chart(fig)
    st.markdown("**Description:** This chart displays the gender distribution across various medical conditions.")

    # Age Distribution Across Patients - Combined Bar and Line Chart
    st.subheader("Age Distribution Across Patients")
    age_counts = filtered_df['Age'].value_counts().sort_index()
    age_df = age_counts.reset_index()
    age_df.columns = ['Age', 'Count']
    fig = px.bar(age_df, x='Age', y='Count', title='Age Distribution Across Patients', color='Count',
                 color_continuous_scale='Blues')
    fig.update_traces(hovertemplate='Age: %{x}<br>Count: %{y}')

    # Adding Line Chart
    fig.add_scatter(x=age_df['Age'], y=age_df['Count'], mode='lines+markers', name='Age Trend',
                    line=dict(color='red'))
    st.plotly_chart(fig)
    st.markdown(
        "**Description:** This chart combines a bar chart and line chart to show the distribution of patient ages and the trend across different age groups.")

with tab2:
    st.header("Financial Insights")

    # Distribution of Billing Amount - Density Plot
    st.subheader("Distribution of Billing Amount")
    fig = px.density_heatmap(filtered_df, x='Billing Amount', title="Density Plot of Billing Amount",
                             color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_traces(hovertemplate='Billing Amount: %{x:.2f}<br>Density: %{z:.2f}')
    st.plotly_chart(fig)
    st.markdown("**Description:** This density plot visualizes the distribution and density of billing amounts.")

    # Average Billing Amount by Hospital
    st.subheader("Average Billing Amount by Hospital")
    avg_billing_by_hospital = filtered_df.groupby("Hospital")["Billing Amount"].mean().reset_index()
    fig = px.bar(avg_billing_by_hospital, x='Hospital', y='Billing Amount',
                 title='Average Billing Amount by Hospital',
                 color='Billing Amount', color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_traces(hovertemplate='%{x}: $%{y:.2f}')
    st.plotly_chart(fig)
    st.markdown("**Description:** This bar chart shows the average billing amount per hospital.")

with tab3:
    st.header("Medical Condition Analysis")

    # Distribution of Medical Conditions - Interactive Bar Chart
    st.subheader("Distribution of Medical Conditions")
    condition_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('Medical Condition', sort='-y'),
        y='count()',
        color=alt.Color('Medical Condition', scale=alt.Scale(scheme='set3')),
        tooltip=['Medical Condition', 'count()']
    ).interactive()
    st.altair_chart(condition_chart, use_container_width=True)
    st.markdown(
        "**Description:** This bar chart depicts the distribution of various medical conditions among the patients.")

    # Length of Stay Analysis - Interactive Line Chart
    st.subheader("Length of Stay Analysis")
    line_chart_data = filtered_df.groupby('Date of Admission')['Length of Stay'].mean().reset_index()
    fig = px.line(line_chart_data, x='Date of Admission', y='Length of Stay', title="Average Length of Stay Over Time",
                  color_discrete_sequence=['#FFA07A'])
    fig.update_traces(hovertemplate='Date: %{x}<br>Length of Stay: %{y} days')
    st.plotly_chart(fig)
    st.markdown("**Description:** This line chart illustrates the average length of stay for patients over time.")

with tab4:
    st.header("Correlation Analysis")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = filtered_df[['Age', 'Billing Amount', 'Length of Stay']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_traces(hovertemplate='%{z:.2f}')
    st.plotly_chart(fig)
    st.markdown("**Description:** This heatmap shows the correlation between age, billing amount, and length of stay.")

    # Scatter Plot of Age vs Billing Amount
    st.subheader("Age vs Billing Amount by Gender")
    fig = px.scatter(filtered_df, x='Age', y='Billing Amount', color='Gender',
                     title="Age vs Billing Amount by Gender",
                     labels={"Age": "Patient Age", "Billing Amount": "Total Billing Amount"},
                     color_discrete_map={"Male": "#1f77b4", "Female": "#ff7f0e"},
                     hover_data=['Age', 'Billing Amount', 'Gender'])
    st.plotly_chart(fig)
    st.markdown(
        "**Description:** This scatter plot visualizes the relationship between patients' age and their billing amount, categorized by gender.")
