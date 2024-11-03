import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# Example: Save feature importance to JSON
feature_importance.to_json("feature_importance.json", orient="records")


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




##
# Create a new feature: point difference
data['point_diff'] = data['UCSD PTS: (Points)'] - data['OPPONONET PTS: (Points)']

# Define features (X) and target (y)
X = data[['UCSD PTS: (Points)', 'OPPONONET PTS: (Points)', 'point_diff', 'UCSD TO: (Turnovers)', 
          'Home/Away  (1 if home, 0 if away)']]  # Features
y = data['win']  # Target


##
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


##

# Check if your dataset includes the required columns; if not, you may need to collect that data.
data['possessions'] = data['UCSD FGA'] - data['UCSD OR'] + data['UCSD TO'] + 0.475 * data['UCSD FTA']
data['offensive_efficiency'] = data['UCSD PTS: (Points)'] / data['possessions']
data['defensive_efficiency'] = data['OPPONONET PTS: (Points)'] / data['possessions']
##

# Calculate turnover differential
data['turnover_diff'] = data['UCSD TO: (Turnovers)'] - data['OPPONENT TO: (Turnovers)']

##

# Compute rolling average of points for the last 5 games
data['ucsd_points_rolling_avg'] = data['UCSD PTS: (Points)'].rolling(window=5).mean()

# Similarly, you could add more rolling metrics for turnovers or opponent points.
data['turnover_rolling_avg'] = data['UCSD TO: (Turnovers)'].rolling(window=5).mean()

##

# Update your features (X) to include new metrics
X = data[['UCSD PTS: (Points)', 'OPPONONET PTS: (Points)', 'point_diff', 'UCSD TO: (Turnovers)',
          'Home/Away  (1 if home, 0 if away)', 'offensive_efficiency', 'defensive_efficiency', 
          'turnover_diff', 'ucsd_points_rolling_avg', 'turnover_rolling_avg']]

y = data['win']

# Split the data and train as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


##
# Display feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print(feature_importance)
