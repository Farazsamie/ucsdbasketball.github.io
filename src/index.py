import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


#Load in UCSD Dataset 
try: 
    data =pd.read_excel('ucsdball.xlsx')
except FileNotFoundError:
    print("File Not Found. Check FIle Path")
    exit ()


#Rename coluums in needed for easier handling 
data.rename(columns={
    'UCSD PTS: (Points)': 'UCSD_PTS',
    'OPPONENT PTS: (Points)': 'Opponent_PTS',
    'Home/Away (1 if hime, 0 if away)': 'Home_Away',
    'Win/Loss (1 if UCSD wins, 0 if UCSD loses)': 'Win,'
}, inplace=True )


#Create "Trailing at Q1?" and "Trailing at Half?"
data['Trailing_Q1'] = (data['UCSD_PTS'] < data['Opponent_PTS']).astype(int)
data['Trailing_Half'] = data.groupby('Date')['Trailing_Q1'].shift(1).fillna(0)

# Win Percentage when Trailing
win_when_trailing_q1 = data[data['Trailing_Q1'] == 1]['Win'].mean() * 100
win_when_trailing_half = data[data['Trailing_Half'] == 1]['Win'].mean() * 100
win_overall = data['Win'].mean() * 100

# Print out results
print(f"Overall Win %: {win_overall:.2f}%")
print(f"Win % when Trailing at Q1: {win_when_trailing_q1:.2f}%")
print(f"Win % when Trailing at Half: {win_when_trailing_half:.2f}%")

# Visualization: Bar Chart for Win % Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=["Overall", "Trailing Q1", "Trailing Half"], 
            y=[win_overall, win_when_trailing_q1, win_when_trailing_half], 
            palette=["green", "red", "orange"])
plt.ylim(0, 100)
plt.ylabel("Win Percentage (%)")
plt.title("UCSD Win % When Trailing vs. Overall")
plt.show()

# Line Chart: Performance Trend When Trailing
plt.figure(figsize=(10, 5))
sns.lineplot(x=data.index, y=data['UCSD_PTS'], label="UCSD Points", color='blue')
sns.lineplot(x=data.index, y=data['Opponent_PTS'], label="Opponent Points", color='red')
plt.title("UCSD Performance Over Games (Trailing vs. Leading)")
plt.xlabel("Games")
plt.ylabel("Points Scored")
plt.legend()
plt.show()