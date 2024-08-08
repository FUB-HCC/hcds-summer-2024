import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd
from multiprocessing import Process
import fairness_functions as ff


def load_data():
    
    data = pd.read_csv('aw_fb_data.csv')

  
    unwanted_columns = ["Unnamed: 0",'X1']
    data = data.drop(columns=unwanted_columns)
    data['device'].replace(['apple watch', 'fitbit'],
                        [0, 1], inplace=True)

    target_column = 'activity'  


    data[target_column].replace(['Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs', 'Running 5 METs', 'Running 7 METs'],
                        [0, 1, 2, 3, 4, 5], inplace=True)

    
    balanced_data = data.groupby(target_column).apply(lambda x: x.sample(min(len(x), 50))).reset_index(drop=True)

  
    X = balanced_data.drop(target_column, axis=1)
    y = balanced_data[target_column]
    return X, y
X,y = load_data()


models = {
    "RandomForest": joblib.load("RandomForest.joblib"),
    "LogisticRegression": joblib.load("LogisticRegression.joblib"),
    "DecisionTree": joblib.load("DecisionTree.joblib"),
    "NeuralNetwork": joblib.load("NeuralNetwork.joblib")
}


def run_dashboard(model_name, port):
    generate_fairness_results(model_name, X, y)
    model = models[model_name]
    explainer = ClassifierExplainer(model, X, y)
    explainer.plot_confusion_matrix(binary=False)
    dashboard = ExplainerDashboard(explainer, title=f"{model_name} Explainer")
    dashboard.run(port=port)

def generate_fairness_results(model_name, X, y):
    y.replace(['Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs', 'Running 5 METs', 'Running 7 METs'],
                        [0, 0, 1, 1, 1, 1], inplace=True)
    model = joblib.load(f'fairness_{model_name}.joblib')
    all_preds = model.predict(X)
    final_data = X.copy()
    final_data['activity'] = y.copy()
    final_data['Prediction'] = all_preds.copy()

    probability_female_with_dynamic_activity = ff.group_fairness(final_data,"gender", 0, "Prediction", 1)
    probability_male_with_dynamic_activity = ff.group_fairness(final_data,"gender", 1, "Prediction", 1)
    probability_female_with_static_activity = ff.group_fairness(final_data,"gender", 0, "Prediction", 0)
    probability_male_with_static_activity = ff.group_fairness(final_data,"gender", 1, "Prediction", 0)
  
    probability_female_with_device_fitbit_and_dynamic_activity = ff.conditional_statistical_parity(final_data, "gender", 0, "Prediction", 1, "device", 1)
    probability_male_with_device_fitbit_and_dynamic_activity = ff.conditional_statistical_parity(final_data, "gender", 1, "Prediction", 1, "device", 1)
    probability_female_with_device_fitbit_and_static_activity = ff.conditional_statistical_parity(final_data, "gender", 0, "Prediction", 0, "device", 1)
    probability_male_with_device_fitbit_and_static_activity = ff.conditional_statistical_parity(final_data, "gender", 1, "Prediction", 0, "device", 1)
    probability_female_with_device_apple_watch_and_dynamic_activity = ff.conditional_statistical_parity(final_data, "gender", 0, "Prediction", 1, "device", 0)
    probability_male_with_device_apple_watch_and_dynamic_activity = ff.conditional_statistical_parity(final_data, "gender", 1, "Prediction", 1, "device", 0)
    probability_female_with_device_apple_watch_and_static_activity = ff.conditional_statistical_parity(final_data, "gender", 0, "Prediction", 0, "device", 0)
    probability_male_with_device_apple_watch_and_static_activity = ff.conditional_statistical_parity(final_data, "gender", 1, "Prediction", 0, "device", 0)
    
    predictive_parity = ff.predictive_parity(final_data, "gender", 0, "Prediction", "activity")
    error_rate = ff.fp_error_rate_balance(final_data, "gender", 0, "Prediction", "activity")

    fairness_result = {
        'probability_female_with_dynamic_activity': [probability_female_with_dynamic_activity],
        'probability_male_with_dynamic_activity': [probability_male_with_dynamic_activity],
        'probability_female_with_static_activity': [probability_female_with_static_activity], 
        'probability_male_with_static_activity': [probability_male_with_static_activity],
        'probability_female_with_device_fitbit_and_dynamic_activity': [probability_female_with_device_fitbit_and_dynamic_activity], 
        'probability_male_with_device_fitbit_and_dynamic_activity': [probability_male_with_device_fitbit_and_dynamic_activity],
        'probability_female_with_device_fitbit_and_static_activity': [probability_female_with_device_fitbit_and_static_activity],
        'probability_male_with_device_fitbit_and_static_activity': [probability_male_with_device_fitbit_and_static_activity],
        'probability_female_with_device_apple_watch_and_dynamic_activity': [probability_female_with_device_apple_watch_and_dynamic_activity], 
        'probability_male_with_device_apple_watch_and_dynamic_activity': [probability_male_with_device_apple_watch_and_dynamic_activity],
        'probability_female_with_device_apple_watch_and_static_activity': [probability_female_with_device_apple_watch_and_static_activity],
        'probability_male_with_device_apple_watch_and_static_activity': [probability_male_with_device_apple_watch_and_static_activity],
        'predictive_parity': [predictive_parity], 
        'error_rate': [error_rate]
    }
    pd.DataFrame(fairness_result).to_csv(f'{model_name}_fairness_result.csv', index=False)



if __name__ == "__main__":
    ports = {
        "RandomForest": 8054,
        "LogisticRegression": 8051,
        "DecisionTree": 8052,
        "NeuralNetwork": 8053
    }
    
    processes = []
    for model_name, port in ports.items():
        p = Process(target=run_dashboard, args=(model_name, port))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
