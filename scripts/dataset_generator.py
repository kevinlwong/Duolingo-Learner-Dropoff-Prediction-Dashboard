# import pandas as pd
# import numpy as np

# # Load the initial dataset
# df = pd.read_csv("/mnt/data/ml_features_extended_v2_rounded.csv")

# # Adjust the 'Dropped_Off' probability based on 'Lessons_Per_Week', 'Avg_Time_Per_Lesson', and 'Quiz_Scores'
# # Define conditions for drop-off probability
# high_lessons = df['Lessons_Per_Week'] > 7
# low_lessons = df['Lessons_Per_Week'] <= 7
# high_time = df['Avg_Time_Per_Lesson'] > 1
# low_time = df['Avg_Time_Per_Lesson'] <= 1
# high_quiz_score = df['Quiz_Scores'] >= 50
# low_quiz_score = df['Quiz_Scores'] < 50

# # Define more realistic 'Dropped_Off' using conditions
# df['Dropped_Off'] = np.where(
#     (low_lessons & high_time & low_quiz_score), 1,  # High drop-off probability if struggling
#     np.where((high_lessons & low_time & high_quiz_score), 0,  # Low drop-off if dedicated
#              np.random.choice([0, 1], size=len(df), p=[0.6, 0.4]))  # Random with bias to lower drop-off
# )

# # Adjust 'Quiz_Scores' to align with 'Lessons_Per_Week' and 'Avg_Time_Per_Lesson'
# df['Quiz_Scores'] = np.where(
#     high_lessons & low_time,  # Dedicated users with high lessons and low time
#     np.random.randint(70, 101, size=len(df)),  # Higher scores
#     np.where(
#         low_lessons & high_time,  # Struggling users with low lessons and high time
#         np.random.randint(0, 60, size=len(df)),  # Lower scores
#         df['Quiz_Scores']  # Keep original for others
#     )
# )

# # Round relevant columns for better consistency
# df['Lessons_Per_Week'] = df['Lessons_Per_Week'].round(2)
# df['Avg_Time_Per_Lesson'] = df['Avg_Time_Per_Lesson'].round(2)

# # Save the adjusted dataset
# output_path = "/mnt/data/ml_features_realistic_v3.csv"
# df.to_csv(output_path, index=False)

# output_path




import os
import pandas as pd
import numpy as np

# Function to generate realistic user data
def generate_realistic_user_data(start_id, end_id, random_seed=42):
    np.random.seed(random_seed)
    
    user_ids = [f"User_{i}" for i in range(start_id, end_id+1)]
    lessons_per_week = np.random.randint(1, 30, size=len(user_ids))  # Integer values for lessons
    avg_time_per_lesson = np.round(np.random.uniform(0.1, 5, size=len(user_ids)), 2)  # Float values
    streaks = np.random.randint(1, 50, size=len(user_ids))  # Integer streak values
    quiz_scores = np.round(np.random.uniform(1, 100, size=len(user_ids)), 2)  # Float values for quiz scores

    # Logic for 'Dropped_Off' based on performance
    dropoff_probability = np.where(
        (lessons_per_week > 15) & (avg_time_per_lesson < 2) & (quiz_scores > 70),
        0.2,  # Less likely to drop off if high performance
        0.8   # More likely to drop off if lower performance
    )
    dropped_off = np.random.binomial(1, dropoff_probability, size=len(user_ids))

    # Create DataFrame
    user_data = pd.DataFrame({
        'User_ID': user_ids,
        'Lessons_Per_Week': lessons_per_week,
        'Avg_Time_Per_Lesson': avg_time_per_lesson,
        'Streaks': streaks,
        'Quiz_Scores': quiz_scores,
        'Dropped_Off': dropped_off
    })
    
    return user_data

# Generate data for user IDs from 401 to 1400
realistic_user_data = generate_realistic_user_data(401, 1400)

# Save to CSV
file_path = './data/ml_features_realistic_v4.csv'
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

realistic_user_data.to_csv(file_path, index=False)

print(f"Data saved successfully to {file_path}")


realistic_user_data.head()