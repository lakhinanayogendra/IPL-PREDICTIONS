import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"C:\Users\YOGENDRA\OneDrive\Desktop\ipldataset.csv")
match_team_runs = df.groupby(['mid', 'batting_team'])['total'].max().reset_index()
match_pivot = match_team_runs.pivot(index='mid', columns='batting_team', values='total').fillna(0)
winners = []
for mid, row in match_pivot.iterrows():
    if row.sum() == 0:
        winner = 'No result'
    else:
        winner = row.idxmax()
    winners.append({'mid': mid, 'winner': winner})
winners_df = pd.DataFrame(winners)
match_base = df.groupby('mid').first().reset_index()
final_df = match_base.merge(winners_df, on='mid')
final_df = final_df[['batting_team', 'bowling_team', 'venue', 'winner']]
final_df = final_df.dropna()
label_encoders = {}
for col in ['batting_team', 'bowling_team', 'venue', 'winner']:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col])
    label_encoders[col] = le
X = final_df[['batting_team', 'bowling_team', 'venue']]
y = final_df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
def predict_winner(batting_team, bowling_team, venue):
    try:
        batting_enc = label_encoders['batting_team'].transform([batting_team])[0]
        bowling_enc = label_encoders['bowling_team'].transform([bowling_team])[0]
        venue_enc = label_encoders['venue'].transform([venue])[0]
        input_data = np.array([[batting_enc, bowling_enc, venue_enc]])
        pred_encoded = model.predict(input_data)[0]
        predicted_team = label_encoders['winner'].inverse_transform([pred_encoded])[0]
        return predicted_team
    except ValueError as e:
        return f"Input error: {e}"
example_batting_team = 'Kolkata Knight Riders'
example_bowling_team = 'Rajasthan Royals'
example_venue = label_encoders['venue'].inverse_transform([final_df['venue'].iloc[0]])[0]
print(f"Match: {example_batting_team} vs {example_bowling_team}")
print(f"Venue: {example_venue}")
print("Predicted Winner:", predict_winner(example_batting_team, example_bowling_team, example_venue))