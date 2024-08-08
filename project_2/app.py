import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from messages import msg
from sklearn.metrics import classification_report
import shap
from explainerdashboard import ClassifierExplainer
from viz import visualize
import fairness_functions as ff
from sklearn.utils import resample

viz = visualize()

# Useful global stuff
DATA_PATH = './data/heart_disease.csv'
DIAGNOSIS_DICT = {0: 'healthy',
                  1: 'ill'}
MODELS_SEED = 123

assert msg.FEATURES_DESC.keys() == msg.FEATURES_VALS.keys()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Data load
df_raw = pd.read_csv(DATA_PATH)  # This has all rows
df_raw['num'] = df_raw['num'].apply(lambda x: 1 if x > 0 else x)
df = df_raw.dropna()  # This has no rows with missing values

# Header
st.title(msg.MAIN_TITLE)
st.write(msg.APP_DESCRIPTION)

# Tabs setup
tab_data, tab_model, tab_xai, tab_fairness = st.tabs(
    ['Exploring Data', 'Building Models', 'Explaining Algorithms', 'Evaluating Fairness'])

# ================
# === SIDE BAR ===
# ================

sidebar = st.sidebar
sidebar.header(msg.FILTER_TITLE)

# Age slider
age = sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()),
                     (int(df['age'].min()), int(df['age'].max())))

# Sex selectbox
sex = sidebar.selectbox('Sex', ('Both', 'Male', 'Female'))

# Diagnosis multiselect
diagnosis = sidebar.multiselect('Diagnosis', df['num'].unique(), df['num'].unique(),
                                format_func=lambda x: DIAGNOSIS_DICT[round(x)])

# Filtering the dataframe
sex_set = {0} if sex == 'Female' else {1} if sex == 'Male' else {0, 1}
filtered_df = df[(df['age'] >= age[0]) & (df['age'] <= age[1]) & (df['num'].isin(diagnosis)) \
                 & df['sex'].isin(sex_set)]

# Count the filtered data
sidebar.write('Total filtered entries: ' + str(filtered_df.shape[0]))
sidebar.write('Fraction of filtered entries: ' + str(round(filtered_df.shape[0] / df.shape[0] * 100, 2)) + '%')

# Filter subsection ends
sidebar.write(msg.SEPARATOR)

# Dataset features info
sidebar.header(msg.FEATURE_TITLE)
sidebar.write(msg.SELECT_FEATURE_MSG)
selectbox_choices = list(df.columns)
trivia_target = sidebar.selectbox("Select Feature", selectbox_choices)
if trivia_target in msg.FEATURES_DESC.keys() and trivia_target in msg.FEATURES_VALS.keys():
    sidebar.write('🤓 Info on *' + trivia_target + '* 🤓')
    sidebar.write('***Description***: ' + msg.FEATURES_DESC[trivia_target])
    sidebar.write('***Possible values***: ' + msg.FEATURES_VALS[trivia_target])
else:
    sidebar.write(msg.MISSING_FEATURE_ERROR)

# Dataset features subsection ends
sidebar.write(msg.SEPARATOR)

# Dataset general info
sidebar.header(msg.DATA_DESCRIPTION_TITLE)
for line in msg.DATASET_DESCRIPTION:
    sidebar.write(line)

X_new = df.drop("num", axis=1)
y_new = df["num"]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=123,
                                                                    stratify=y_new)


@st.cache_resource
def train_rf_model(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=123)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return rf_model, pd.DataFrame(report).transpose()


@st.cache_resource
def train_lr_model(X_train, y_train, X_test, y_test):
    lr_model = LogisticRegression(max_iter=1000, random_state=123)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return lr_model, pd.DataFrame(report).transpose()


# Train the models on the new dataset
rf_model, rf_report_df = train_rf_model(X_train_new, y_train_new, X_test_new, y_test_new)
lr_model, lr_report_df = train_lr_model(X_train_new, y_train_new, X_test_new, y_test_new)

# ================
# === DATA TAB ===
# ================


with tab_data:
    # Full dataframe
    st.write(msg.FULL_DATASET_TITLE)
    st.write(msg.FULL_DATASET_MSG)
    st.dataframe(df_raw)

    # Filtered dataframe
    st.write(msg.FILTERED_DATASET_TITLE)
    st.write(msg.FITERED_DATASET_MSG)

    if filtered_df.shape[0]:
        st.dataframe(filtered_df)
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

    # Individual distributions
    CONTINUOUS = ('age', 'trestbps', 'chol', 'thalach', 'oldpeak')  # These features will need a histogram
    st.write(msg.FEATURE_DISTRIBUTION_TITLE)
    st.write(msg.FEATURE_DISTRIBUTION_MSG)

    if filtered_df.shape[0]:
        feature_name = st.selectbox("Select Feature", df.columns, key='individual_distr')
        if feature_name in CONTINUOUS:
            st.pyplot(visualize().continuous_distr(filtered_df, feature_name))
        else:
            st.pyplot(visualize().categorical_distr(filtered_df, feature_name))
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

    # Mutual distributions
    st.write(msg.FEATURE_CORRELATION_TITLE)
    st.write(msg.FEATURE_CORRELATION_MSG)
    if filtered_df.shape[0]:
        feature1_name = st.selectbox("Select Feature 1", df.columns, index=0)
        feature2_name = st.selectbox("Select Feature 2", df.columns, index=1)

        # If names are equal - just put out the marginal distribution
        if feature1_name == feature2_name:
            if feature1_name in CONTINUOUS:
                st.pyplot(visualize().continuous_distr(filtered_df, feature1_name))
            else:
                st.pyplot(visualize().categorical_distr(filtered_df, feature1_name))
        # If names are not equal - several options
        else:
            # Both features continuous -> show scatterplot
            if feature1_name in CONTINUOUS and feature2_name in CONTINUOUS:
                st.pyplot(visualize().scatterplot(filtered_df, feature1_name, feature2_name))
            # One continuous, one categorical -> show boxplot
            elif feature1_name in CONTINUOUS:
                st.pyplot(visualize().boxplot(filtered_df, feature1_name, feature2_name))
            elif feature2_name in CONTINUOUS:
                st.pyplot(visualize().boxplot(filtered_df, feature2_name, feature1_name))
            # Both features categorical -> show heatmap?
            else:
                st.pyplot(visualize().heatmap(filtered_df, feature1_name, feature2_name))
    else:
        st.write(msg.EMPTY_FILTER_ERROR)

# ==================
# === MODELS TAB ===
# ==================
with tab_model:
    st.header("Building And Explaining Models")
    st.write(msg.MODELS_DESCRIPTION[0])
    st.markdown(
        msg.MODELS_WARNING,
        unsafe_allow_html=True
    )

    # Model selection
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"], key="model_choice")

    if model_choice == "Random Forest":
        st.subheader("Random Forest Model")
        st.write(msg.MODEL_FOREST_DESCRIPTION)
        st.write("Results of the model training")
        st.write(rf_report_df)
        # Plot AUROC
        fig_auroc = viz.plot_auroc(rf_model, X_test_new, y_test_new, 'Random Forest')
        st.pyplot(fig_auroc)
    
        # Plot Confusion Matrix
        fig_cm = viz.plot_confusion_matrix(y_test_new, rf_model.predict(X_test_new), 'Random Forest')
        st.pyplot(fig_cm)

    elif model_choice == "Logistic Regression":
        st.subheader("Logistic Regression Model")
        st.write(msg.MODEL_LOGREG_DESCRIPTION)
        st.write("Results of the model training")
        st.write(lr_report_df)
        
        # Plot AUROC
        fig_auroc = viz.plot_auroc(lr_model, X_test_new, y_test_new, 'Logistic Regression')
        st.pyplot(fig_auroc)
    
        # Plot Confusion Matrix
        fig_cm = viz.plot_confusion_matrix(y_test_new, lr_model.predict(X_test_new), 'Logistic Regression')
        st.pyplot(fig_cm)


# ===============
# === XAI TAB ===
# ===============
with tab_xai:
    st.header(msg.MODELS_TITLE)
    st.write(msg.XAI_DESCRIPTION)
    model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        st.subheader(msg.MODEL_FOREST_TITLE)
        st.write(msg.XAI_RF_DESCRIPTION)
        explainer = ClassifierExplainer(rf_model, X_test_new, y_test_new, shap_kwargs=dict(approximate=True))
        fi = explainer.get_importances_df()
        fi_sorted = fi.sort_values(by='MEAN_ABS_SHAP', ascending=False)
        fi_sorted['Feature'] = pd.Categorical(fi_sorted['Feature'], categories=fi_sorted['Feature'], ordered=True)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='MEAN_ABS_SHAP', y='Feature', data=fi_sorted, orient='h')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.ylabel('Feature')
        plt.title('Feature Importances')

        st.pyplot(plt)

        st.markdown(
            msg.SHAP_VALUES_DISCUSSION,
            unsafe_allow_html=True
        )

        st.write(msg.SHAP_VALUES_INDIVIDUAL_DISCUSSION)
        shap_values = explainer.get_shap_values_df()
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values.values, X_test_new, show=False)
        st.pyplot(fig)

        st.write(msg.XAI_RF_GLOBAL_DISCUSSION)
        st.write(msg.XAI_BOOST_CA_DISCUSSION)
        st.write(msg.XAI_BOOST_FBS_DISCUSSION)

    elif model_choice == "Logistic Regression":

        for line in msg.XAI_LOGREG_DESCRIPTION:
            if line[:len(msg.FORMULA_TOKEN)] == msg.FORMULA_TOKEN:
                st.latex(line[len(msg.FORMULA_TOKEN):])
            else:
                st.write(line)
        coef_df = pd.DataFrame({
            'Feature': X_new.columns,
            'Coefficient': lr_model.coef_[0]
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        fig, ax = plt.subplots()
        coef_df.plot(kind='barh', x='Feature', y='Coefficient', ax=ax)
        ax.set_title(msg.FEATURE_COEFFICIENTS)
        st.pyplot(fig)

        st.write(msg.XAI_LOGREG_DISCUSSION)

# ====================
# === FAIRNESS TAB ===
# ====================
pred_df = df.copy() 
X = df.drop("num", axis=1)
pred_df["prediction_lr"] = lr_model.predict(X)
pred_df["prediction_rf"] = rf_model.predict(X)

with tab_fairness:
    for line in msg.FAIRNESS_DESCRIPTION:
        st.write(line)

    st.markdown(msg.FF_METRICS_INTRODUCTION)

    st.write(msg.FAIRNESS_LOGREG_TITLE)
    st.write(msg.FAIRNESS_LOGREG_MSG)
    pf = ff.group_fairness(pred_df, 'sex', 0, 'prediction_lr', 1)
    pm = ff.group_fairness(pred_df, 'sex', 1, 'prediction_lr', 1)
    pfs = ff.conditional_statistical_parity(pred_df, "sex", 0, "prediction_lr", 1, "fbs", 1)
    pms = ff.conditional_statistical_parity(pred_df, "sex", 1, "prediction_lr", 1, "fbs", 1)
    ppvf = ff.predictive_parity(pred_df, "sex", 0, "prediction_lr", "num")
    ppvm = ff.predictive_parity(pred_df, "sex", 1, "prediction_lr", "num")
    fprf = ff.fp_error_rate_balance(pred_df, "sex", 0, "prediction_lr", "num")
    fprm = ff.fp_error_rate_balance(pred_df, "sex", 1, "prediction_lr", "num")

    fairness_metrics_lr = {
        "No.": [1,2,3, 4],
        "Metric": [
            msg.FF_METRIC_1,
            msg.FF_METRIC_2,
            msg.FF_METRIC_3, 
            msg.FF_METRIC_4,
        ],
        "Female": [pf, pfs, ppvf, fprf],
        "Male": [pm, pms, ppvm, fprm]
    }

    fairness_df_lr = pd.DataFrame(fairness_metrics_lr)
    fairness_df_lr = fairness_df_lr.set_index('No.')
    st.dataframe(fairness_df_lr)

    st.write(msg.FAIRNESS_RF_TITLE)
    st.write(msg.FAIRNESS_RF_MSG)
    pf = ff.group_fairness(pred_df, 'sex', 0, 'prediction_rf', 1)
    pm = ff.group_fairness(pred_df, 'sex', 1, 'prediction_rf', 1)
    pfs = ff.conditional_statistical_parity(pred_df, "sex", 0, "prediction_rf", 1, "fbs", 1)
    pms = ff.conditional_statistical_parity(pred_df, "sex", 1, "prediction_rf", 1, "fbs", 1)
    ppvf = ff.predictive_parity(pred_df, "sex", 0, "prediction_rf", "num")
    ppvm = ff.predictive_parity(pred_df, "sex", 1, "prediction_rf", "num")
    fprf = ff.fp_error_rate_balance(pred_df, "sex", 0, "prediction_rf", "num")
    fprm = ff.fp_error_rate_balance(pred_df, "sex", 1, "prediction_rf", "num")

    fairness_metrics_rf = {
        "No.": [1,2,3, 4],
        "Metric": [
            msg.FF_METRIC_1,
            msg.FF_METRIC_2,
            msg.FF_METRIC_3, 
            msg.FF_METRIC_4,
        ],
        "Female": [pf, pfs, ppvf, fprf],
        "Male": [pm, pms, ppvm, fprm]
    }

    fairness_df_rf = pd.DataFrame(fairness_metrics_rf)
    fairness_df_rf = fairness_df_rf.set_index('No.')
    st.dataframe(fairness_df_rf)

    st.write(msg.FAIRNESS_DISCUSS_TITLE)
    st.write(msg.FAIRNESS_DISCUSSION)

    # Balancingg using over sampling starts here
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['num'] = df_raw['num'].apply(lambda x: 1 if x > 0 else x)
    df = df_raw.dropna()

    # Balance the dataset based on 'sex'
    df_male = df[df['sex'] == 1]
    df_female = df[df['sex'] == 0]

    # Determine which class to oversample
    if len(df_male) < len(df_female):
        df_minority = df_male
        df_majority = df_female
    else:
        df_minority = df_female
        df_majority = df_male

    # Oversample minority class
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=MODELS_SEED)

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Prepare the balanced dataset for modeling
    X_balanced = df_balanced.drop("num", axis=1)
    y_balanced = df_balanced["num"]

    # Split the balanced dataset
    X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=MODELS_SEED, stratify=y_balanced
    )

    # Train the models on the balanced dataset
    rf_model_balanced, rf_report_df_balanced = train_rf_model(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced)
    lr_model_balanced, lr_report_df_balanced = train_lr_model(X_train_balanced, y_train_balanced, X_test_balanced, y_test_balanced)

    # Update the predictions for the entire balanced dataset
    pred_df_balanced = df_balanced.copy()
    X = df_balanced.drop("num", axis=1)
    pred_df_balanced["prediction_lr"] = lr_model_balanced.predict(X)
    pred_df_balanced["prediction_rf"] = rf_model_balanced.predict(X)

    st.write("We have balanced the dataset based on the 'sex' feature using random oversampling and re-trained the models and evaluated the fairness metrics again. Let's review the fairness metrics for our balanced models.")
    st.write(msg.FAIRNESS_LOGREG_BALANCED_TITLE)
    pf = ff.group_fairness(pred_df_balanced, 'sex', 0, 'prediction_lr', 1)
    pm = ff.group_fairness(pred_df_balanced, 'sex', 1, 'prediction_lr', 1)
    pfs = ff.conditional_statistical_parity(pred_df_balanced, "sex", 0, "prediction_lr", 1, "fbs", 1)
    pms = ff.conditional_statistical_parity(pred_df_balanced, "sex", 1, "prediction_lr", 1, "fbs", 1)
    ppvf = ff.predictive_parity(pred_df_balanced, "sex", 0, "prediction_lr", "num")
    ppvm = ff.predictive_parity(pred_df_balanced, "sex", 1, "prediction_lr", "num")
    fprf = ff.fp_error_rate_balance(pred_df_balanced, "sex", 0, "prediction_lr", "num")
    fprm = ff.fp_error_rate_balance(pred_df_balanced, "sex", 1, "prediction_lr", "num")

    fairness_metrics_lr_balanced = {
        "No.": [1,2,3, 4],
        "Metric": [
            msg.FF_METRIC_1,
            msg.FF_METRIC_2,
            msg.FF_METRIC_3, 
            msg.FF_METRIC_4,
        ],
        "Female": [pf, pfs, ppvf, fprf],
        "Male": [pm, pms, ppvm, fprm]
    }

    fairness_df_lr_balanced = pd.DataFrame(fairness_metrics_lr_balanced)
    fairness_df_lr_balanced = fairness_df_lr_balanced.set_index('No.')
    st.dataframe(fairness_df_lr_balanced)

    st.write(msg.FAIRNESS_RF_BALANCED_TITLE)
    pf = ff.group_fairness(pred_df_balanced, 'sex', 0, 'prediction_rf', 1)
    pm = ff.group_fairness(pred_df_balanced, 'sex', 1, 'prediction_rf', 1)
    pfs = ff.conditional_statistical_parity(pred_df_balanced, "sex", 0, "prediction_rf", 1, "fbs", 1)
    pms = ff.conditional_statistical_parity(pred_df_balanced, "sex", 1, "prediction_rf", 1, "fbs", 1)
    ppvf = ff.predictive_parity(pred_df_balanced, "sex", 0, "prediction_rf", "num")
    ppvm = ff.predictive_parity(pred_df_balanced, "sex", 1, "prediction_rf", "num")
    fprf = ff.fp_error_rate_balance(pred_df_balanced, "sex", 0, "prediction_rf", "num")
    fprm = ff.fp_error_rate_balance(pred_df_balanced, "sex", 1, "prediction_rf", "num")

    fairness_metrics_rf_balanced = {
        "No.": [1,2,3, 4],
        "Metric": [
            msg.FF_METRIC_1,
            msg.FF_METRIC_2,
            msg.FF_METRIC_3, 
            msg.FF_METRIC_4,
        ],
        "Female": [pf, pfs, ppvf, fprf],
        "Male": [pm, pms, ppvm, fprm]
    }

    fairness_metrics_rf_balanced = pd.DataFrame(fairness_metrics_rf_balanced)
    fairness_metrics_rf_balanced = fairness_metrics_rf_balanced.set_index('No.')
    st.dataframe(fairness_metrics_rf_balanced)

    st.write(msg.FAIRNESS_BALANCED_DISCUSS_TITLE)
    st.write(msg.FAIRNESS_DISCUSSION_BALANCED)

