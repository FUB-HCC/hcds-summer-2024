import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import joblib

def load_data():

    data = pd.read_csv('aw_fb_data.csv')

    unwanted_columns = ["Unnamed: 0",'X1']
    data = data.drop(columns=unwanted_columns)
    data['device'].replace(['apple watch', 'fitbit'],
                        [0, 1], inplace=True)
    target_column = 'activity'  

    data[target_column].replace(['Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs', 'Running 5 METs', 'Running 7 METs'],
                        [0, 1, 2, 3, 4, 5], inplace=True)


    balanced_data = data.groupby(target_column).apply(lambda x: x.sample(min(len(x), 500))).reset_index(drop=True)

    X = balanced_data.drop(target_column, axis=1)
    y = balanced_data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(max_iter=200, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}.joblib")
