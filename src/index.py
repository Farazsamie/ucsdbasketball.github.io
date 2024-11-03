import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load your Excel file
data = pd.read_excel('ucsdball.xlsx')

# Take a quick look at the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Remove rows with missing data
data = data.dropna()

# Add a 'win' column, where 1 = UCSD win and 0 = UCSD loss
data['win'] = data['Win/Loss  (1 if UCSD wins, 0 if UCSD loses)']

# Create a new feature: point difference
data['point_diff'] = data['UCSD PTS: (Points)'] - data['OPPONONET PTS: (Points)']

# Define features (X) and target (y)
X = data[['UCSD PTS: (Points)', 'OPPONONET PTS: (Points)', 'point_diff', 'UCSD TO: (Turnovers)', 
          'Home/Away  (1 if home, 0 if away)']]  # Features
y = data['win']  # Target

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
