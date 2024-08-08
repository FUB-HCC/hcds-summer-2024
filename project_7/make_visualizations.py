import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import confusion_matrix, RocCurveDisplay, classification_report
import seaborn as sns
import joblib


DATASET_PATH =  "asthma_disease_data.csv"
MODEL_PATH = "SVM_linear_scaled_p.sav"
OUTPUT_PATH = "dataset_enriched.csv"

def load_data(path: str)->pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df


def load_model(path: str):
    model = pickle.load((open(path, 'rb')))
    return model

def add_predictions(data: pd.DataFrame, model)->pd.DataFrame:
    X = data.drop([ 'Diagnosis', 'PatientID', 'DoctorInCharge'], axis=1)
    scaler = joblib.load('scaler.pkl')
    X_scaled = scaler.transform(X)
    Y_pred = model.predict(X_scaled)
    data['Prediction'] = Y_pred
    return data


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10,8))
    class_names = ['No Asthma', 'Asthma']
    sns.heatmap(cm, annot=True,  cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    return fig


def plot_roc(Y_pred, Y_true):
    fig, ax = plt.subplots(figsize=(8, 8))

    RocCurveDisplay.from_predictions(
        Y_true,
        Y_pred,
        name=f"ROC curve for Asthma classifier",
        ax=ax,
    )
    _ = ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Receiver Operating Characteristic Curve",
    )
    plt.legend()
    return fig



data = load_data(DATASET_PATH)
model = load_model(MODEL_PATH)
data = add_predictions(data, model)
data.to_csv(OUTPUT_PATH)
#data = pd.read_csv("data/dataset_enriched.csv")
#confusion_m = plot_confusion_matrix(data['Diagnosis'], data['Prediction'])
#report = classification_report(data['Diagnosis'], data['Prediction'], output_dict=True)
#print(report["0"], type(report["0"]))