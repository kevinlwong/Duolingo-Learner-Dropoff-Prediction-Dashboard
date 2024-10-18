import os
import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate a random timestamp
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds())),
    )

# Date range for the 'Last_Activity' column
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)

# Initialize an empty list to store the user data
data = []

# Generate data for Users from 101 to 1400
for i in range(101, 1401):
    user_id = f"User_{i}"
    
    # Generate realistic values for each column
    lessons_completed = random.randint(1, 100)  # Number of lessons completed (realistic range)
    time_spent_per_lesson = round(random.uniform(5, 60), 1)  # Time per lesson in minutes
    quiz_scores = random.randint(1, 100)  # Quiz scores between 1 and 100
    last_activity = random_date(start_date, end_date).strftime("%m/%d/%Y %H:%M")  # Random timestamp
    # Drop-off likelihood increases with low lessons, low scores, or high time spent
    if lessons_completed < 30 and time_spent_per_lesson > 40 and quiz_scores < 50:
        dropped_off = 1
    else:
        dropped_off = 0
    
    # Append the generated data for the current user
    data.append([user_id, lessons_completed, time_spent_per_lesson, quiz_scores, last_activity, dropped_off])

# Convert the list to a DataFrame
columns = ['User_ID', 'Lessons_Completed', 'Time_Spent_Per_Lesson', 'Quiz_Scores', 'Last_Activity', 'Dropped_Off']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
file_path = './data/realistic_learner_data.csv'
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the generated data to a CSV file
df.to_csv('./data/realistic_learner_data.csv', index=False)

print("Dataset generated and saved to 'realistic_learner_data.csv'.")
