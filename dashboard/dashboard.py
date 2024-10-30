import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
import pandasql as ps
import sqlite3

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

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Learner Dashboard", "Machine Learning Features", "SQL Query"])

with tab1:
    st.header("Project Overview")
    st.write("""
        Welcome to Kevin Wong's **Duolingo Learner Dashboard**! This application is designed to analyze and predict learner behavior on the Duolingo platform. 
        By leveraging machine learning and data visualization, it helps in identifying factors that contribute to user engagement and drop-off.
    """)
    
    st.subheader("Key Sections")
    st.write("""
        **1. Learner Dashboard:**
        - **Purpose:** Provides a high-level overview of user activity with customizable filters.
        - **Features:**
            - **Date Range**: Select specific periods to view user data.
            - **Drop-Off Status**: Filter users based on whether they remained active or dropped off.
            - **User ID**: Analyze individual user behaviors.
        - **Visualizations**: Includes bar charts, pie charts, and line charts for data like lessons completed, retention over time, and quiz score distributions.
        
        **2. Machine Learning Features:**
        - **Purpose:** Analyze learner engagement and predict drop-off probability using machine learning.
        - **Features:**
            - **Filter Options**: Filter by lessons per week.
            - **Model Comparison**: Evaluate and compare machine learning models using metrics such as accuracy, ROC curves, and confusion matrices.
            - **Interactive Predictions**: Input data and predict drop-off likelihood using trained models.
        
        **3. SQL Query:**
        - **Purpose:** Run custom SQL queries on the dataset for advanced data exploration.
        - **Features:** Input SQL queries to explore user behavior and model features in greater detail.
    """)
    
    st.subheader("Navigation Instructions")
    st.write("""
        - **Adjust Filters:** Use the sidebar to filter data by date range, drop-off status, and user ID in both the Learner Dashboard and Machine Learning Features tabs.
        - **View Visualizations:** Charts and metrics will update based on your filter selections. Hover over elements for more information.
        - **Run SQL Queries:** In the SQL Query tab, enter your custom SQL query and click “Run Query” to retrieve insights from the data.
    """)
    
    st.subheader("Additional Features")
    st.write("""
        - **Tooltips and Explanations:** Consider adding tooltips to explain specific terms or features.
        - **Download Options:** You could download filtered data or query results as CSV files.
        - **Expandable Visualizations:** Use expandable sections for charts and graphs to view details and switch between visualization types.
    """)


# Learner Dashboard section
with tab2:


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
    

        # 3. Filter by Drop-Off Status (Checkbox)
        st.sidebar.subheader("Filter by Drop-Off Status")
        show_dropped_off = st.sidebar.checkbox("Show Only Dropped Off Users", False)

        # Apply drop-off filter
        if show_dropped_off:
            filtered_data = filtered_data[filtered_data['Dropped_Off'] == 1]

        st.sidebar.subheader("Filter by User ID")
        user_ids = filtered_data['User_ID'].unique()
        selected_users = st.sidebar.multiselect("Select User ID(s)", user_ids, default=user_ids)

        # Apply user ID filter
        filtered_data = filtered_data[filtered_data['User_ID'].isin(selected_users)]


        # ---- MAIN CONTENT ----
        # Convert 'User_ID' to numeric format by removing the 'User_' prefix and changing to int
        data['User_ID'] = data['User_ID'].str.extract(r'(\d+)').astype(int)

        # Now 'User_ID' is purely numeric, so sorting will work correctly
        data = data.sort_values(by='User_ID').reset_index(drop=True)

        # Show the filtered dataset
        st.write("Here is a snapshot of the mock user data:")
        # st.dataframe(filtered_data)
        
        st.dataframe(data, width=1000)
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
# Machine Learning Features section
# elif tab == "Machine Learning Features":
with tab3:

    st.title("Duolingo Machine Learning Prediction Dashboard")

    st.header("ML Features Section")

    # Load your ML features data
    ml_data_path = '../data/ml_features_realistic_v3.csv'  # Use your ML dataset path
    ml_data = pd.read_csv(ml_data_path)



    # Sidebar Filters
    st.sidebar.header("Filter Options for Machine Learning Data")

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
    st.write("Here is a snapshot of the mock filtered ML features data:")
    st.dataframe(filtered_ml_data, width=1000)






    # Machine Learning Model Building and Evaluation
    # Remove the 'User_' prefix from the 'User_ID' column
    ml_data['User_ID'] = ml_data['User_ID'].str.replace('User_', '')

    # Convert 'User_ID' to integer, if desired, for ordering purposes
    ml_data['User_ID'] = ml_data['User_ID'].astype(int)

    print(ml_data.head())

    X = ml_data.drop('Dropped_Off', axis=1)  # Features
    y = ml_data['Dropped_Off']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    #Train and evaluate each model and make predictions
    predictions = {}
    accuracy_scores = []
    conf_matrices = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        # Calculate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracy_scores.append({'Model': name, 'Accuracy': accuracy})

        #collect data for roc curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr,tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        #collect confusion matrix data
        conf_matrices[name] = confusion_matrix(y_test, predictions[name])

    # st.title('Model Comparison for Predicting Drop-Off Status')

    # # Accuracy Bar Chart
    # st.subheader("Model Accuracy")
    # accuracy_df = pd.DataFrame(accuracy_scores)
    # accuracy_chart = alt.Chart(accuracy_df).mark_bar().encode(
    #     x='Model',
    #     y='Accuracy',
    #     color='Model'
    # ).properties(width=600, height=500)
    # st.altair_chart(accuracy_chart, use_container_width=True)


    Modeltab, ROCtab, Conftab, ClassTab, DataTab = st.tabs(["Model Comparison", "ROC Curves", "Confusion Matrices", "Classification Reports", "Charts and Insights"])

    with Modeltab:
        st.title('Model Comparison for Predicting Drop-Off Status')

        # Accuracy Bar Chart
        st.subheader("Model Accuracy")
        accuracy_df = pd.DataFrame(accuracy_scores)
        accuracy_chart = alt.Chart(accuracy_df).mark_bar().encode(
            x='Model',
            y='Accuracy',
            color='Model'
        ).properties(width=600, height=600)
        st.altair_chart(accuracy_chart, use_container_width=True)

    with ROCtab:
        col1, col2 = st.columns(2)
        # Display ROC Curves
        # st.subheader("ROC Curves")
        # Display ROC Curves, alternating between columns
        for i, (name, roc) in enumerate(roc_data.items()):
            # Alternate between col1 and col2
            with col1 if i % 2 == 0 else col2:
                st.write(f"**{name} (AUC = {roc['auc']:.2f})**")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.plot(roc['fpr'], roc['tpr'], label=f"{name} (AUC = {roc['auc']:.2f})")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)

    with Conftab:
        col1, col2 = st.columns(2)
        # Display ROC Curves
        # st.subheader("ROC Curves")
        # Display ROC Curves, alternating between columns
        for i, (name, cm) in enumerate(conf_matrices.items()):
            # Alternate between col1 and col2
            with col1 if i % 2 == 0 else col2:
            # st.write(f"**{name}**")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Active', 'Dropped Off'], yticklabels=['Active', 'Dropped Off'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f"{name}")
                st.pyplot(fig)

    with ClassTab:
        # Display Classification Reports
        # st.subheader("Classification Reports")
        # for name, y_pred in predictions.items():
            # st.write(f"**{name}**")
            # st.text(classification_report(y_test, y_pred, target_names=['Active', 'Dropped Off']))
        # Create a container for neat organization
        with st.container():
            # Display each model's result in a column
            # cols = st.columns(len(predictions))  # Adjust the number of columns based on the number of models
            
            # Iterate over models and display their results in columns
            for i, (name, y_pred) in enumerate(predictions.items()):
                # Select the appropriate column based on index
                # with cols[i]:
                    st.subheader(f"{name}")
                    accuracy = next(item['Accuracy'] for item in accuracy_scores if item['Model'] == name)
                    st.write(f"**Accuracy Score:** {accuracy:.2f}")
                    
                    # Create a neatly formatted classification report
                    report = classification_report(y_test, y_pred, target_names=['Active', 'Dropped Off'], output_dict=True)
                    # Convert the dictionary to a DataFrame
                    report_df = pd.DataFrame(report)

                    # Print the columns to see their names
                    # st.write("Columns in the report DataFrame:", report_df.columns)

                    # Drop the 'support' column if it exists
                    if 'support' in report_df.columns:
                        report_df = report_df.drop(columns='support')
                    else:
                        # 'support' is likely present in the index instead
                        report_df = report_df.drop(index='support', errors='ignore')
                    # Remove the accuracy key
                    if 'accuracy' in report:
                        del report['accuracy']
                    report_df = pd.DataFrame(report).transpose()
                    st.write("Classification Report:")
                    st.dataframe(report_df, width=1000)  # Use container width for better visibility

                # Display the Predictions DataFrame for this model
                    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            # with cola:
                # st.write(f"Predictions DataFrame for {name}:")
                # st.write(predictions_df.head(5))
                # Provide an option to expand and view all rows
                    with st.expander("View all rows for Predictions Dataframe"):
                        st.write(predictions_df, width=1000)

        

    st.sidebar.header("Input new data for prediction")
    input_data = {}

    # Specify which columns should be treated as integers
    integer_columns = ['User_ID', 'Lessons_Per_Week', 'Streaks']

    for col in X.columns:
        if col in integer_columns:
            # Treat as integer input
            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=int(X[col].mean()), format="%d")
        else:
            # Treat as float input
            input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(X[col].mean()), format="%.2f")
    input_data = {key: [value] for key, value in input_data.items()}

    #make predictions on the new data
    new_data = pd.DataFrame(input_data)

    st.sidebar.header("Predictions on New Data")
    for name, model in models.items():
        prediction = model.predict(new_data)[0]
        prediction_text = "Dropped Off" if prediction == 1 else "Active"
        st.sidebar.write(f"{name}: {prediction_text}")


    with DataTab:
        # --- Additional Charts and Insights ---

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

        print(ml_data.columns)


with tab4:
    st.header("SQL Query")
    # Section for interactive sql queries
    # Streamlit UI for SQL Query Input
    st.header("Interactive SQL Query on Data")
    st.subheader("Enter your SQL query below to interact with the data:")
    query_example = """SELECT User_ID, 
        Lessons_Per_Week, 
        Avg_Time_Per_Lesson, 
        Quiz_Scores, 
        Streaks, 
        Dropped_Off
    FROM ml_data 
    WHERE Lessons_Per_Week > 5 
    AND Quiz_Scores > 50 
    GROUP BY User_ID 
    HAVING AVG(Avg_Time_Per_Lesson) < 2 
    ORDER BY Streaks DESC;
    """
    # Text area for SQL query input
    query = st.text_area("SQL Query", query_example, height=300)

    # Button to execute the query
    if st.button("Run Query"):
        try:
            # Run the SQL query on the data DataFrame
            query_result = ps.sqldf(query, locals())
            st.write("Query Results:")
            st.dataframe(query_result)  # Display the query result in a table
        except Exception as e:
            st.error(f"Error executing query: {e}")




# You can add more charts, metrics, or insights from the ML features below...

