import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Set the page layout to wide, but control the max width using custom CSS
st.set_page_config(page_title="Duolingo Learner Dashboard", layout="wide")

[theme]
primaryColor = "#0000FF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F0F0"
textColor = "#000000"

# CSS for customizing scrollbar
st.markdown(
    """
    <style>
  

    /* Width of the scrollbar */
    ::-webkit-scrollbar {
        width: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title for your dashboard
st.title("Duolingo Learner Dashboard with Controlled Width")

# Load your data (replace with actual data path)
data_path = '../data/learner_data.csv'  # Use your dataset path
data = pd.read_csv(data_path)

# Convert 'Last_Activity' to datetime format
data['Last_Activity'] = pd.to_datetime(data['Last_Activity'])

# Sidebar Filters
st.sidebar.header("Filter Options")

# 1. Filter by Date Range
st.sidebar.subheader("Filter by Date Range")
date_min = data['Last_Activity'].min().to_pydatetime()  # Convert to Python datetime
date_max = data['Last_Activity'].max().to_pydatetime()  # Convert to Python datetime
start_date, end_date = st.sidebar.slider(
    "Select Date Range",
    min_value=date_min,
    max_value=date_max,
    value=(date_min, date_max),
    format="MM/DD/YY"
)

# Apply date range filter
filtered_data = data[(data['Last_Activity'] >= start_date) & (data['Last_Activity'] <= end_date)]


# Check if filtered data is empty
if filtered_data.empty:
    st.warning("No data available for the selected date range.")
else:
    # 2. Filter by User ID (Multiselect)
    st.sidebar.subheader("Filter by User ID")
    user_ids = filtered_data['User_ID'].unique()
    selected_users = st.sidebar.multiselect("Select User ID(s)", user_ids, default=user_ids)

    # Apply user ID filter
    filtered_data = filtered_data[filtered_data['User_ID'].isin(selected_users)]

    # 3. Filter by Drop-Off Status (Checkbox)
    st.sidebar.subheader("Filter by Drop-Off Status")
    show_dropped_off = st.sidebar.checkbox("Show Only Dropped Off Users", False)

    # Apply drop-off filter
    if show_dropped_off:
        filtered_data = filtered_data[filtered_data['Dropped_Off'] == 1]

    # ---- MAIN CONTENT ----
    # Show the filtered dataset
    st.write("Here is a snapshot of the learner data:")
    st.dataframe(filtered_data)

    # First Row: Bar chart (Lessons Completed) and Pie chart (Drop-Off Status) with moderate width differences
    col1, col2 = st.columns([1.5, 1])  # First column 1.5x wider than second

    # Bar chart for Lessons Completed in col1
    with col1:
        st.subheader("Lessons Completed by Users")
        fig, ax = plt.subplots(figsize=(8, 4))  # Adjust figure size to make it smaller
        ax.bar(filtered_data['User_ID'], filtered_data['Lessons_Completed'])
        ax.set_xlabel("User ID")
        ax.set_ylabel("Lessons Completed")
        ax.set_title("Lessons Completed by User")
        ax.set_xticks(ax.get_xticks()[::10])  # Display every 10th user ID to reduce clutter
        ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right')
        st.pyplot(fig)

    # Pie chart for Drop-Off Status in col2
    with col2:
        st.subheader("Drop-Off Status of Learners")
        fig, ax = plt.subplots(figsize=(4, 5))  # Adjust pie chart size
        drop_off_counts = filtered_data['Dropped_Off'].value_counts()
        # ax.pie(drop_off_counts, labels=['Active', 'Dropped Off'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # ax.set_title("Proportion of Users Dropped Off")
        # st.pyplot(fig)
        # Dynamically set labels based on available data
        labels = []
        colors = []

        if 0 in drop_off_counts.index:
            labels.append("Active")
            colors.append("green")
        if 1 in drop_off_counts.index:
            labels.append("Dropped Off")
            colors.append("red")

        # Draw the pie chart with dynamic labels and colors
        fig, ax = plt.subplots()
        ax.pie(drop_off_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title("Proportion of Users Dropped Off")
        st.pyplot(fig)

    # Second Row: Line chart (User Retention) and Histogram (Quiz Scores) with equal width
    col3, col4 = st.columns([1, 1])  # Equal column widths

    # Line chart for User Retention in col3
    with col3:
        st.subheader("User Retention Over Time")
        user_retention = filtered_data.groupby(filtered_data['Last_Activity'].dt.to_period("M")).size()
        fig, ax = plt.subplots(figsize=(6, 3))  # Adjust size to reduce width
        ax.plot(user_retention.index.to_timestamp(), user_retention.values)
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Users")
        ax.set_title("User Retention Over Time")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B \n%Y'))
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    # Histogram for Quiz Scores in col4
    with col4:
        st.subheader("Quiz Scores Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size to make it smaller
        ax.hist(filtered_data['Quiz_Scores'], bins=10, color='blue', alpha=0.7)
        ax.set_xlabel("Quiz Scores")
        ax.set_ylabel("Number of Users")
        ax.set_title("Distribution of Quiz Scores")
        st.pyplot(fig)

    # Third Row: Metrics (with equal widths)
    col5, col6, col7 = st.columns([1, 1, 1])  # All columns equal width

    # Metric 1: Average Quiz Score
    with col5:
        avg_quiz_score = filtered_data['Quiz_Scores'].mean()
        st.metric(label="Average Quiz Score", value=f"{avg_quiz_score:.2f}")

    # Metric 2: Drop-Off Rate
    with col6:
        drop_off_rate = filtered_data['Dropped_Off'].mean() * 100
        st.metric(label="Overall Drop-Off Rate", value=f"{drop_off_rate:.2f}%")

    # Metric 3: Average Lessons Completed
    with col7:
        avg_lessons_completed = filtered_data['Lessons_Completed'].mean()
        st.metric(label="Average Lessons Completed", value=f"{avg_lessons_completed:.2f}")




# -------------------- NEW SECTION FOR ML FEATURES --------------------
st.header("ML Features Section")

# Load your ML features data
ml_data_path = '../data/ml_features.csv'  # Use your ML dataset path
ml_data = pd.read_csv(ml_data_path)

# Sidebar Filters
st.sidebar.header("Filter Options")

# 1. Filter by Lessons Per Week
st.sidebar.subheader("Filter by Lessons Per Week")

# Find the min and max values of "Lessons Per Week" to use for the slider
lessons_min = int(ml_data['Lessons_Per_Week'].min())
lessons_max = int(ml_data['Lessons_Per_Week'].max())

# Create a range slider to filter the "Lessons Per Week"
lessons_range = st.sidebar.slider(
    "Select Lessons Per Week Range",
    min_value=lessons_min,
    max_value=lessons_max,
    value=(lessons_min, lessons_max)  # Default to the full range
)

# Apply the filter to the data
filtered_ml_data = ml_data[(ml_data['Lessons_Per_Week'] >= lessons_range[0]) & (ml_data['Lessons_Per_Week'] <= lessons_range[1])]
# Display ML features data as a table (filtered)
st.write("Here is a snapshot of the filtered ML features data:")
st.dataframe(filtered_ml_data)

# Define two columns for the charts
col1, col2 = st.columns(2)

# --- Chart 1: Box Plot with Altair ---
with col1:
    st.subheader("Lessons Per Week vs. Drop-Off Status")
    box_plot = alt.Chart(filtered_ml_data).mark_boxplot().encode(
        x=alt.X('Dropped_Off:O', title='Drop-Off Status (0 = Active, 1 = Dropped Off)'),
        y=alt.Y('Lessons_Per_Week:Q', title='Lessons Per Week'),
        color='Dropped_Off:N'
    ).properties(
        width=300,
        title="Lessons Per Week Distribution by Drop-Off Status"
    )
    st.altair_chart(box_plot, use_container_width=True)

# --- Chart 2: Bar Chart of Average Metrics ---
with col2:
    avg_time_spent = filtered_ml_data['Avg_Time_Per_Lesson'].mean()
    avg_quiz_score = filtered_ml_data['Quiz_Scores'].mean()
    avg_streaks = filtered_ml_data['Streaks'].mean()

    metrics_data = pd.DataFrame({
        'Metrics': ['Avg Time Spent Per Lesson', 'Avg Quiz Score', 'Avg Streaks Maintained'],
        'Values': [avg_time_spent, avg_quiz_score, avg_streaks]
    })

    st.subheader("Comparison of Avg Time Spent Per Lesson, Quiz Score, and Streaks Maintained")
    bar_chart = alt.Chart(metrics_data).mark_bar().encode(
        x=alt.X('Metrics', sort=None, title='Metrics'),
        y=alt.Y('Values', title='Average Values'),
        color=alt.Color('Metrics', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Metrics', 'Values']
    ).properties(
        width=300,
        title="Average Time Spent Per Lesson, Quiz Scores, and Streaks"
    )
    st.altair_chart(bar_chart, use_container_width=True)

# Additional Two Altair Charts in Two Columns Below the Above Charts
col3, col4 = st.columns(2)

# --- Chart 3: Additional Bar Chart of Average Metrics ---
with col3:
    alt_bar_chart = alt.Chart(metrics_data).mark_bar().encode(
        x=alt.X('Metrics', sort=None, title='Metrics'),
        y=alt.Y('Values', title='Average Values'),
        color=alt.Color('Metrics', scale=alt.Scale(scheme='tableau10')),
        tooltip=['Metrics', 'Values']
    ).properties(
        width=300,
        title="Average Time Spent Per Lesson, Quiz Scores, and Streaks (Altair)"
    )
    st.altair_chart(alt_bar_chart, use_container_width=True)

# --- Chart 4: Scatter Plot for Lessons Per Week vs. Quiz Scores ---
with col4:
    scatter_chart = alt.Chart(filtered_ml_data).mark_circle(size=60).encode(
        x=alt.X('Lessons_Per_Week', title='Lessons Per Week'),
        y=alt.Y('Quiz_Scores', title='Quiz Scores'),
        color=alt.Color('Dropped_Off:N', legend=alt.Legend(title="Drop-Off Status")),
        tooltip=['Lessons_Per_Week', 'Quiz_Scores', 'Dropped_Off']
    ).properties(
        width=300,
        title='Lessons Per Week vs. Quiz Scores'
    )
    st.altair_chart(scatter_chart, use_container_width=True)


# You can add more charts, metrics, or insights from the ML features below...

