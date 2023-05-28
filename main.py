import mysql.connector
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json

def get_data(year):
    year_changed = year.replace("/", "_")
    data = json.load(open('db.json'))
    mydb = mysql.connector.connect(
        host=data["host"],
        user=data["user"],
        password=data["password"],
        port=data["port"],
        database=data["database"],
    )
    mycursor = mydb.cursor(dictionary=True)
    mycursor.execute(f"SELECT AVG(DEF) AS RPDEF FROM Player WHERE SeasonYear = %s AND DEF != 0 GROUP BY Team ORDER BY Team", (year,))
    ratings = mycursor.fetchall()
    for rating in ratings:
        X.append(rating["RPDEF"])

    team_defenses = pd.read_csv('./team-defenses/team_defenses_' + year_changed + '.csv')
    team_defenses = team_defenses.sort_values(by='Team', ascending=True)
    y.extend(team_defenses["DEF"].values.tolist())

X = []
y = []
years = ["13/14", "14/15", "15/16", "16/17", "17/18", "18/19", "19/20", "20/21", "21/22", "22/23"]
for year in years:
    get_data(year)
X = np.array(X).reshape((-1, 1))
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R^2 Score:', r2)

defensive_rating = 4.458445896657303  # Example defensive rating
predicted_points = model.predict([[defensive_rating]])
print('Predicted Average Points Received:', predicted_points[0])

