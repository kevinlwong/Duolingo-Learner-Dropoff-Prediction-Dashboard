import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Set the page layout to wide, but control the max width using custom CSS
st.set_page_config(page_title="Duolingo Learner Dashboard", layout="wide")

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
st.title("Duolingo Learner Dashboard with Mock Data")

# Load your data (replace with actual data path)
data_path = '../data/learner_data.csv'  # Use your dataset path
data = pd.read_csv(data_path)

# Convert 'Last_Activity' to datetime format
data['Last_Activity'] = pd.to_datetime(data['Last_Activity'])

# sidebar image
st.sidebar.image('../images/duolingo_wave.png', width=275)
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
    # Convert 'User_ID' to numeric format by removing the 'User_' prefix and changing to int
    data['User_ID'] = data['User_ID'].str.extract('(\d+)').astype(int)

    # Now 'User_ID' is purely numeric, so sorting will work correctly
    data = data.sort_values(by='User_ID').reset_index(drop=True)

    # Show the filtered dataset
    st.write("Here is a snapshot of the learner data:")
    # st.dataframe(filtered_data)
    
    st.dataframe(data)
    # Convert 'User_ID' to a string to treat it as a categorical field
    data['User_ID'] = data['User_ID'].astype(str)

    # Explicitly sort 'User_ID' by the order they appear in the dataset
    user_id_order = data['User_ID'].tolist()

    # Filtered data (apply your filters as needed)
    filtered_data = data  # Assuming you've applied any necessary filters before this
    user_id_list = filtered_data['User_ID'].tolist()
    every_fifth_user_id = user_id_list[::5]  # Slices list to every fifth element
    # First Row: Bar chart (Lessons Completed) and Pie chart (Drop-Off Status) with moderate width differences
    col1, col2 = st.columns([1.5 , 1])  
    # 1. Bar Chart for Lessons Completed (in col1)
    with col1:
        st.subheader("Lessons Completed by Users")
        lessons_completed_chart = alt.Chart(filtered_data).mark_bar(color='#7AC70C').encode(
            x=alt.X('User_ID:N', title='User ID', axis=alt.Axis(labelAngle=-45, values=every_fifth_user_id), sort=user_id_order),
            y=alt.Y('Lessons_Completed:Q', title='Lessons Completed'),
            tooltip=['User_ID', 'Lessons_Completed']
        ).properties(
            width=350,
            height=300,
            title='Lessons Completed by User'
        )
        st.altair_chart(lessons_completed_chart, use_container_width=True)

    # 2. Pie Chart for Drop-Off Status (in col2)
    with col2:
        st.subheader("Drop-Off Status of Learners")
        drop_off_counts = filtered_data['Dropped_Off'].value_counts().reset_index()
        drop_off_counts.columns = ['Dropped_Off', 'Count']
        drop_off_counts['Label'] = drop_off_counts['Dropped_Off'].map({0: "Active", 1: "Dropped Off"})
        drop_off_counts['Color'] = drop_off_counts['Dropped_Off'].map({0: "green", 1: "red"})

        pie_chart = alt.Chart(drop_off_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Label", type="nominal", scale=alt.Scale(domain=["Active", "Dropped Off"], range=["green", "red"])),
            tooltip=['Label', 'Count']
        ).properties(
            width=200,
            height=300,
            title="Proportion of Users Dropped Off"
        )
        st.altair_chart(pie_chart, use_container_width=True)

    # Define the second row of columns
    col3, col4 = st.columns(2)

    # 3. Line Chart for User Retention Over Time (in col3)
    with col3:
        st.subheader("User Retention Over Time")
        user_retention = filtered_data.groupby(filtered_data['Last_Activity'].dt.to_period("M")).size().reset_index()
        user_retention.columns = ['Month', 'Count']
        user_retention['Month'] = user_retention['Month'].dt.to_timestamp()

        retention_chart = alt.Chart(user_retention).mark_line(point=True, color='#7AC70C').encode(
            x=alt.X('Month:T', title="Month"),
            y=alt.Y('Count:Q', title="Number of Users"),
            tooltip=['Month', 'Count']
        ).properties(
            width=350,
            height=300,
            title="User Retention Over Time"
        )
        st.altair_chart(retention_chart, use_container_width=True)

    # 4. Histogram for Quiz Scores (in col4)
    with col4:
        st.subheader("Quiz Scores Distribution")
        quiz_histogram = alt.Chart(filtered_data).mark_bar(color='#7AC70C').encode(
            x=alt.X('Quiz_Scores:Q', bin=alt.Bin(maxbins=10), title="Quiz Scores"),
            y=alt.Y('count()', title="Number of Users"),
            tooltip=['Quiz_Scores', 'count()']
        ).properties(
            width=350,
            height=300,
            title="Distribution of Quiz Scores"
        )
        st.altair_chart(quiz_histogram, use_container_width=True)

    # Define a third row for metrics (in three equal-width columns)
    col5, col6, col7 = st.columns(3)

    # Metrics
    with col5:
        avg_quiz_score = filtered_data['Quiz_Scores'].mean()
        st.metric(label="Average Quiz Score", value=f"{avg_quiz_score:.2f}")

    with col6:
        drop_off_rate = filtered_data['Dropped_Off'].mean() * 100
        st.metric(label="Overall Drop-Off Rate", value=f"{drop_off_rate:.2f}%")

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
        color=alt.Color('Metrics', scale=alt.Scale(scheme='viridis')),
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
        color=alt.Color('Metrics', scale=alt.Scale(scheme='greens')),
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
        color=alt.Color('Dropped_Off:N', legend=alt.Legend(title="Drop-Off Status"), scale=alt.Scale(scheme='tableau20')),
        tooltip=['Lessons_Per_Week', 'Quiz_Scores', 'Dropped_Off']
    ).properties(
        width=300,
        title='Lessons Per Week vs. Quiz Scores'
    )
    st.altair_chart(scatter_chart, use_container_width=True)


# You can add more charts, metrics, or insights from the ML features below...

