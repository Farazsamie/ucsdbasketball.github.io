import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns


# Load the Excel file
try:
    data = pd.read_excel('ucsdball.xlsx')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Preview the dataset
print("Data Preview:")
print(data.head())

# Check and handle missing values
print("\nMissing Values:")
print(data.isnull().sum())
data.fillna(method='ffill', inplace=True)  # Forward fill missing values
data.fillna(method='bfill', inplace=True)  # Backward fill if forward fill doesn't work

# Rename columns for clarity
data.rename(columns={
    'UCSD PTS: (Points)': 'UCSD_PTS',
    'OPPONONET PTS: (Points)': 'Opponent_PTS',
    'UCSD TO: (Turnovers)': 'UCSD_Turnovers',
    'OPPONENT TO: (Turnovers)': 'Opponent_Turnovers',
    'Home/Away  (1 if home, 0 if away)': 'Home_Away',
    'Win/Loss  (1 if UCSD wins, 0 if UCSD loses)': 'Win'
}, inplace=True)

# Feature Engineering
data['win'] = data['Win']  # Binary target variable
data['point_diff'] = data['UCSD_PTS'] - data['Opponent_PTS']
data['possessions'] = data['UCSD FGA'] - data['UCSD OR'] + data['UCSD_Turnovers'] + 0.475 * data['UCSD FTA']
data['offensive_efficiency'] = data['UCSD_PTS'] / data['possessions']
data['defensive_efficiency'] = data['Opponent_PTS'] / data['possessions']
data['turnover_diff'] = data['UCSD_Turnovers'] - data['Opponent_Turnovers']

# Rolling averages
data['ucsd_points_rolling_avg'] = data['UCSD_PTS'].rolling(window=5, min_periods=1).mean()
data['turnover_rolling_avg'] = data['UCSD_Turnovers'].rolling(window=5, min_periods=1).mean()

# Define features and target
X = data[['UCSD_PTS', 'Opponent_PTS', 'point_diff', 'UCSD_Turnovers', 'Home_Away', 
          'offensive_efficiency', 'defensive_efficiency', 'turnover_diff', 
          'ucsd_points_rolling_avg', 'turnover_rolling_avg']]
y = data['win']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model initialization
model = LogisticRegression(class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({'Feature': [
    'UCSD_PTS', 'Opponent_PTS', 'point_diff', 'UCSD_Turnovers', 'Home_Away', 
    'offensive_efficiency', 'defensive_efficiency', 'turnover_diff', 
    'ucsd_points_rolling_avg', 'turnover_rolling_avg'], 
    'Coefficient': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save feature importance to JSON
feature_importance.to_json("feature_importance.json", orient="records", indent=4)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Save the model for future use
joblib.dump(model, "basketball_model.pkl")
print("\nModel saved as 'basketball_model.pkl'.")

# Save the scaler for consistent preprocessing
joblib.dump(scaler, "scaler.pkl")
print("\nScaler saved as 'scaler.pkl'.")
