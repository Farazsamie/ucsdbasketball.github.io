import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
try:
    data = pd.read_excel('ucsdball.xlsx')

except FileNotFoundError:
    print("File Was Not found")
    exit()
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Preview the dataset
print("Data Preview:")
print(data.head())

# Feature Engineering
data['win'] = data['Win/Loss  (1 if UCSD wins, 0 if UCSD loses)']  
data['point_diff'] = data['UCSD PTS: (Points)'] - data['OPPONONET PTS: (Points)']
data['possessions'] = data['UCSD FGA'] - data['UCSD OR'] + data['UCSD TO: (Turnovers)'] + 0.475 * data['UCSD FTA']
data['offensive_efficiency'] = data['UCSD PTS: (Points)'] / data['possessions']
data['defensive_efficiency'] = data['OPPONONET PTS: (Points)'] / data['possessions']
data['turnover_diff'] = data['UCSD TO: (Turnovers)'] - data['OPPONENT TO: (Turnovers)']

# Rolling averages
data['ucsd_points_rolling_avg'] = data['UCSD PTS: (Points)'].rolling(window=5, min_periods=1).mean()
data['turnover_rolling_avg'] = data['UCSD TO: (Turnovers)'].rolling(window=5, min_periods=1).mean()

# Define features and target
X = data[['UCSD PTS: (Points)', 'OPPONONET PTS: (Points)', 'point_diff', 'UCSD TO: (Turnovers)', 
          'Home/Away  (1 if home, 0 if away)', 'offensive_efficiency', 'defensive_efficiency', 
          'turnover_diff', 'ucsd_points_rolling_avg', 'turnover_rolling_avg']]
y = data['win']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Selection & Hyperparameter Tuning
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Neural Network': MLPClassifier(random_state=42)
}

# Hyperparameter grid for each model
param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
}

best_models = {}
best_scores = {}

# Perform hyperparameter tuning
for name, model in models.items():
    print(f"\nTuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_

    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Evaluate the best model on the test set
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# Plot ROC Curve
y_probs = best_model.predict_proba(X_test)[:, 1]  # Get probability scores
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.show()

# Save the best model and scaler
joblib.dump(best_model, "best_basketball_model.pkl")
print("\nBest Model saved as 'best_basketball_model.pkl'.")

joblib.dump(scaler, "scaler.pkl")
print("\nScaler saved as 'scaler.pkl'.")


import json

# Save model performance results
results = {
    "best_model": best_model_name,
    "accuracy": accuracy * 100,
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "roc_auc": roc_auc
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# Save confusion matrix
cm_data = {
    "labels": ["Loss", "Win"],
    "matrix": cm.tolist()
}

with open("confusion_matrix.json", "w") as f:
    json.dump(cm_data, f, indent=4)

print("\nResults saved for web visualization!")