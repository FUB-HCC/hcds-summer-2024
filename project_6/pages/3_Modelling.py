import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fairness_functions as ff
import dalex as dx 
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import warnings
import pickle

plotly.offline.init_notebook_mode()
warnings.filterwarnings("ignore")

with open("models.pkl", "rb") as file:
    # Deserialize and load the object from the file
    models = pickle.load(file)
    
with open("plots.pkl", "rb") as file:
    # Deserialize and load the object from the file
    plots = pickle.load(file)

plotly.offline.init_notebook_mode()
warnings.filterwarnings("ignore")

st.set_page_config(page_title = "Modelling", layout = "wide")

st.title(":brain: Modelling")

h1n1_vaccine_tab, seasonal_vaccine_tab = st.tabs([":microbe: H1N1 Vaccine", ":face_with_thermometer: Seasonal Vaccine"])

explained_dict = {

    "probability_distribution_explained":
        '''
        This plot shows the distribution (i.e. frequency) of all predictions made by the model for $x \\in [0, 1]$ on the test data.
        ''',

    "classification_report_explained":
        '''
        The **Classification Report** provides a detailed breakdown of a classification model's performance. It includes the following metrics for each class:

        1. **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. Precision answers the question: "Of all the instances that the model predicted as positive, how many were actually positive?"

        $$
        \\text{Precision} = \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} + \\text{False Positives (FP)}}
        $$

        2. **Recall (Sensitivity or True Positive Rate)**: The ratio of correctly predicted positive observations to all observations in the actual class. Recall answers the question: "Of all the instances that were actually positive, how many did the model correctly identify?"

        $$
        \\text{Recall} = \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} + \\text{False Negatives (FN)}}
        $$

        3. **F1-Score**: The weighted average of Precision and Recall. The F1-Score conveys the balance between the precision and the recall. A good F1-Score indicates both high precision and high recall.

        $$
        \\text{F1-Score} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}
        $$

        4. **Support**: The number of actual occurrences of the class in the dataset. Support provides context to the other metrics, showing how many instances of each class are in the test set.

        - **Class-specific Metrics**:
        
        - **Precision, Recall, F1-Score for each class** (e.g., class 0 and class 1): These show how well the model performed on each individual class.
        
        - **Support for each class**: Shows the number of true instances for each class in the test dataset.

        - **Accuracy**:
            - The overall accuracy of the model, i.e., the ratio of correctly predicted observations to the total observations.

        - **Macro Average**:
            - The arithmetic mean of precision, recall, and F1-Score for all classes. It treats all classes equally, regardless of their support.

        - **Weighted Average**:
            - The mean of precision, recall, and F1-Score for all classes, weighted by the number of true instances (support) for each class. It takes class imbalance into account.
        ''',

    "confusion_matrix_explained": 
        '''
        A **confusion matrix** is a table used to evaluate the performance of a classification model. It compares the actual target values with the values predicted by the model. Here's how it works:
        
        - **True Positives (TP)**: The model correctly predicts the positive class.
        
        - **True Negatives (TN)**: The model correctly predicts the negative class.
        
        - **False Positives (FP)**: The model incorrectly predicts the positive class (also known as a "Type I error").
        
        - **False Negatives (FN)**: The model incorrectly predicts the negative class (also known as a "Type II error").

        The confusion matrix looks like this:

        |                    | Predicted Negative | Predicted Positive |
        |--------------------|--------------------|--------------------|
        | **Actual Positive**| FN                 | TP                 |
        | **Actual Negative**| TN                 | FP                 |

        This matrix helps you see not just overall accuracy, but also where the model might be making errors.
        ''',
    "roc_curve_explained": 
        '''
        The **ROC (Receiver Operating Characteristic) curve** is a graphical representation of a classification model's performance across all classification thresholds. It plots:

        - **True Positive Rate (TPR)** or Sensitivity on the Y-axis.
        
        - **False Positive Rate (FPR)** on the X-axis.

        The **AUC (Area Under the Curve)** is a single number summary of the ROC curve. It ranges from 0 to 1, where 1 means perfect classification and 0.5 indicates random guessing. Higher AUC values indicate better model performance.
        ''',
    "feature_importance_explained": 
        '''
        **Variable importance** tells us how much each feature (or variable) in the data contributes to the prediction made by the model. In other words, it ranks the features by their impact on the model's predictions. This is useful to understand which features are the most influential and which ones might be less relevant or even redundant.
        The higher the score for a variable, the larger effect it has on the model to predict the target. 
        
        Here, variable importance is computed by the drop-out loss.
        
        $$\\text{Dropout Loss} = \\text{Performance with all variables} − \\text{Performance with variable dropped}$$
        
        A larger dropout loss indicates that the variable is more important because its removal significantly impacts the model's performance.
        ''',
    "partial_dependence_explained": 
        '''
        **Partial Dependence (PD)** shows the relationship between a specific feature and the predicted outcome of the model, while keeping other features constant. It helps to visualize how changes in one feature affect the model's predictions. This is useful for understanding the marginal effect of a feature on the predicted outcome.
        ''',
    "shapley_values_explained": 
        '''
        **Shapley values** are a method from cooperative game theory used to fairly distribute the "payout" among players based on their contributions. In data science, they are used to explain the contribution of each feature to a particular prediction made by the model. Shapley values provide a way to understand and interpret the predictions of complex models by showing how much each feature contributed to the final prediction.
        ''',
    "fairness_check_explained":
        '''           
        Plots for the protected groups:
        
        - Equal opportunity ratio
            - $$\\frac{TP}{TP + FN}$$

        - Predictive parity ratio
            - $$\\frac{TP}{TP + FP}$$
        
        - Predictive equality ratio
            - $$\\frac{FP}{FP + TN}$$

        - Accuracy equality ratio
            - $$\\frac{TP + TN}{TP + FP + TN + FN}$$

        - Statistical parity ratio
            - $$\\frac{TP + FP}{TP + FP + TN + FN}$$
        
        The idea here is that ratios between scores of privileged and unprivileged metrics should be close to 1. The closer the more fair the model is. But to relax this criterion a little bit, it can be written more thoughtfully:
        
        $$\\forall_{i \in \{a,b,...,z\}} \epsilon < \\frac{\\text{metric}_i}{\\text{metric}_\\text{privileged}} < \\frac{1}{\\epsilon}$$
        
        Where the epsilon is a value between 0 and 1, it should be a minimum acceptable value of the ratio. On default, it is 0.8, which adheres to four-fifths rule (80\\% rule) often looked at in hiring.                                  
        
        If a bar reaches the red field, it means that for this metric model is exceeding the (epsilon, 1/epsilon) range.
        ''',
    "fairness_radar_explained":
        '''
        Plot which shows each parity loss of metric in form of point on ploar coordinate system.
        
        Parity loss apart from being an attribute is a way that summarizes the bias across subgroups. Using the following formula:

        $$\\text{metric}_{\\text{parity loss}} = \sum_{i\in\{a,b,...,z\}}|\\log(\\frac{\\text{metric}_i}{\\text{metric}_{\\text{privileged}}})|$$

        It is a function that is symmetrical in terms of division (f(a/b) = f(b/a)). The intuition behind this formula is simple. The bigger the difference in metrics the higher the parity loss will be. 
        '''
}

analysis_dict = {
    "h1n1_vaccine": {
        "prob_dist":
            '''
            The distribution is heavy tailed with a lot of smaller predictions (i.e. < 0.2). This indicates the model largely predicts an individual will not recieve the H1N1 vaccination.
            ''',
        "class_report":
            '''
            The model performs much better on the majority class (i.e. 0 (no vaccination)) with an F-1 score of 0.9. Comparitively, an F-1 score of 0.5 for the minority class (i.e. 1 (vaccination))
            suggests this model does not perform well with such an imbalanced dataset.
            ''',
        "cf":
            '''
            The model does a very good job predicting when a person will not get vaccinated. However, this also leads to "overpredicting" the majority class. As a result, 
            there is a larger proportion of false negative (FN) predictions, where we expect someone not to get vaccinated, but they actually *do* get vaccinated.
            ''',
        "roc_auc":
            '''
            The ROC curve is promising with an AUC value of 0.83. There is room for improvement especially when predicting the minority class.
            ''',
        "vi":
            '''
            - Top 3 most important features used by the model are:

                - Doctor recommending H1N1 vaccine

                - Opinion if the H1N1 vaccine is effective

                - Opinion on the H1N1 being a serious risk

            - Protected characteristics (age, sex, ethnicity, education, income) improve the model a negligible amount. Therefore, it may be worthwhile removing them to mitigate any unfairness/biasdness leaking into the model.

            - Interestingly, behaviour characteristics (e.g. wearing face mask, washing hands, avoiding large gatherings) also improves the model by a negligible amount.

            - Opinion-based characteristics (H1N1 vaccine effectiveness, risk of getting sick without vaccine/from taking vaccine) are by far the most important collection of features used by the model.
            
            - Health characteristics (which includes the most important feature overall doctor recommendation, as well as knowledge and concern about H1N1) are also a significant grouping of variables.
            ''',
        "pd":
            '''
            There are 3 key variables which increase the likelihood an individual gets the vaccine:

            - Doctor recommending H1N1 vaccine
                - When the doctor recommends getting vaccinated, this increases the likelihood an idividual *does* get vaccinated.

            - Opinion if the H1N1 vaccine is effective
                - When an individual believes the vaccine is effective, this increases the likelihood an idividual *does* get vaccinated.

            - Opinion on the H1N1 being a serious risk
                - When an individual believes they are at risk from contracting H1N1, this increases the likelihood an idividual *does* get vaccinated.
            ''',
        "shap_vals_1":
            '''
            Text
            ''',
        "shap_vals_0":
            '''
            Text
            ''',
        "shap_vals_05":
            '''
            Text
            ''',
        "gf":
            '''
            For group fairness between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "gf_1":
            '''
            For group fairness between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "csp":
            '''
            For conditional statistical parity between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "csp_1":
            '''
            For conditional statistical parity between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "pp":
            '''
            For predictive parity between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "pp_1":
            '''
            For predictive parity between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "fperb":
            '''
            For false positive error rate balance between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "fperb_1":
            '''
            For false positive error rate balance between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "western":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Black" and "Hispanic" ehtnicities.
                - "Other or multiple" ethnicities also very close to being regarded as unfairly modelled.

            - Predictive parity ratio
                - No unfairness detected for any of the ethnicity levels.
            
            - Predictive equality ratio
                - Unfairness detected for all of the ethnicity levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the ethnicity levels.

            - Statistical parity ratio
                - Unfairness detected for all of the ethnicity levels.
            ''',
        "western_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            ''',
        "educated":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Missing and "<12 years" education levels.
                - No unfairness detected for "Some College" or "12 years" education levels.

            - Predictive parity ratio
                - No unfairness detected for any of the education levels.
            
            - Predictive equality ratio
                - Unfairness detected for all of the education levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the education levels.

            - Statistical parity ratio
                - Unfairness detected for "Missing", "<12 years" and "12 years" education levels.
                - "Some College" education level also very close to being regarded as unfairly modelled.
            ''',
        "educated_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            ''',
        "rich":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Missing and "Below Poverty" income levels.
                - No unfairness detected for "<= $75,000, above poverty" education levels.

            - Predictive parity ratio
                - No unfairness detected for any of the income levels.
            
            - Predictive equality ratio
                - Unfairness detected for all of the income levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the income levels.

            - Statistical parity ratio
                - Unfairness detected for all of the income levels.
            ''',
        "rich_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            '''
    },
    "seasonal_vaccine": {
        "prob_dist":
            '''
            Distribution of predicted values appears quite uniform, with equal frequencies of predicted values across $x \\in [0, 1]$
            ''',
        "class_report":
            '''
            The model can handle this target feature well, predicting both the negative (0) and positive (1) class accurately with F1-scores of 0.81 and 0.76 respectively.
            ''',
        "cf":
            '''
            The target variable (i.e. seasonal vaccination) appears much more balanced in this dataset and as a result the model can predict between classes fairly accurately.
            There is a slight "overprediction" of False Negatives.
            ''',
        "roc_auc":
            '''
            The ROC curve is promising with an AUC value of 0.86, modelling the data well.
            ''',
        "vi":
            '''
            - Top 3 most important features used by the model are:

                - Opinion on the seasonal flu being a serious risk

                - Doctor recommending seasonal vaccine

                - Opinion if the seasonal vaccine is effective

            - Protected characteristics (age, sex, ethnicity, education, income) improve the model only a small amount. Therefore, it may be worthwhile removing them to mitigate any unfairness/biasdness leaking into the model.

            - Interestingly, behaviour characteristics (e.g. wearing face mask, washing hands, avoiding large gatherings) also improves the model by a negligible amount.

            - Health characteristics (which includes one of the most important feature overall doctor recommendation, as well as knowledge and concern about seasonal flu) are a significant grouping of variables, but do not influence the model as when predicting for H1N1.

            - Opinion-based characteristics (H1N1 vaccine effectiveness, risk of getting sick without vaccine/from taking vaccine) are by far the most important collection of features used by the model.
            ''',
        "pd":
            '''
            There are 3 key variables which increase the likelihood an individual gets the vaccine:

            - Opinion on the seasonal flue being a serious risk
                - When an individual believes they are at risk from contracting H1N1, this increases the likelihood an idividual *does* get vaccinated.

            - Doctor recommending seasonal flu vaccine
                - When the doctor recommends getting vaccinated, this increases the likelihood an idividual *does* get vaccinated.

            - Opinion if the seasonal flu vaccine is effective
                - When an individual believes the vaccine is effective, this increases the likelihood an idividual *does* get vaccinated.
            ''',
        "shap_vals_1":
            '''
            Text
            ''',
        "shap_vals_0":
            '''
            Text
            ''',
        "shap_vals_05":
            '''
            Text
            ''',
        "gf":
            '''
            For group fairness between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "gf_1":
            '''
            For group fairness between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "csp":
            '''
            For conditional statistical parity between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "csp_1":
            '''
            For conditional statistical parity between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "pp":
            '''
            For predictive parity between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "pp_1":
            '''
            For predictive parity between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "fperb":
            '''
            For false positive error rate balance between classes in :red-background[sex], no unfairness/biasdness is detected.
            ''',
        "fperb_1":
            '''
            For false positive error rate balance between classes in :red-background[chronic medical condition], no unfairness/biasdness is detected.
            ''',
        "western":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Black" and "Hispanic" ehtnicities.
                - "Other or multiple" ethnicities also very close to being regarded as unfairly modelled.

            - Predictive parity ratio
                - No unfairness detected for any of the ethnicity levels.
            
            - Predictive equality ratio
                - Unfairness detected for all of the ethnicity levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the ethnicity levels.

            - Statistical parity ratio
                - Unfairness detected for all of the ethnicity levels.
            ''',
        "western_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            ''',
        "educated":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Missing education level.
                - No unfairness detected for "Some College", "12 years", or "<12 years" education levels.

            - Predictive parity ratio
                - No unfairness detected for any of the education levels.
            
            - Predictive equality ratio
                - Unfairness detected for all of the education levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the education levels.

            - Statistical parity ratio
                - Unfairness detected for "Missing and "<12 years" education levels.
                - No unfairness detected for "Some College" or "12 years" education levels.
            ''',
        "educated_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            ''',
        "rich":
            '''
            - Equal opportunity ratio
                - Unfairness detected for "Below Poverty" income levels.
                - No unfairness detected for "Missing", or "<= $75,000, above poverty" education levels.

            - Predictive parity ratio
                - No unfairness detected for any of the income levels.
            
            - Predictive equality ratio
                - Unfairness detected for "Below Poverty" income levels.
                - No unfairness detected for "Missing", or "<= $75,000, above poverty" education levels.

            - Accuracy equality ratio
                -  No unfairness detected for any of the income levels.

            - Statistical parity ratio
                - Unfairness detected for "Below Poverty" income levels.
                - No unfairness detected for "Missing", or "<= $75,000, above poverty" education levels.
            ''',
        "rich_radar":
            '''
            Unfairness detected for FPR, TPR, STP metrics.
            '''
    }
}

model = "h1n1_vaccine"

with h1n1_vaccine_tab:
    h1n1_1, h1n1_2, h1n1_3 = st.tabs([":chart_with_upwards_trend: Model Performance", ":bulb: Model Explanations", ":busts_in_silhouette: Model Fairness"])
    with h1n1_1:

        h1h1_p_dist_1, h1n1_class_report_1 = st.columns([2, 1])
        h1h1_p_dist_2, h1n1_class_report_2 = st.columns([2, 1], vertical_alignment = "center")
        h1h1_p_dist_3, h1n1_class_report_3 = st.columns([2, 1])

        fig = px.histogram(models[model]["predict_proba"][:,1], nbins = 10, title = f"Distribution of Predicted Values for {model}")
        fig.update_traces(marker_line_width = 1, marker_line_color = "black")
        fig.update_layout(xaxis_title = "Probability", yaxis_title = "Count", showlegend = False)
        h1h1_p_dist_1.expander(":interrobang: Distribution of Predicted Values Explained").markdown(explained_dict["probability_distribution_explained"])
        h1h1_p_dist_2.plotly_chart(fig)
        h1h1_p_dist_3.expander(":sparkles: Analysis").markdown(analysis_dict[model]["prob_dist"])

        h1n1_class_report_1.expander(":interrobang: Classification Report Explained").markdown(explained_dict["classification_report_explained"])
        h1n1_class_report_2.dataframe(pd.DataFrame(classification_report(models[model]["y_test"], models[model]["predict"], output_dict = True)).T)
        h1n1_class_report_3.expander(":sparkles: Analysis").markdown(analysis_dict[model]["class_report"])
        
        st.markdown("---")
        h1n1_cf, h1n1_roc = st.columns(2)
        with h1n1_cf:
            confusion_matrix_explained = st.expander(":interrobang: Confusion Matrix Explained")
            with confusion_matrix_explained:
                st.markdown(explained_dict["confusion_matrix_explained"])

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(models[model]["y_test"], models[model]["predict"])).plot(ax = ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["cf"])

        with h1n1_roc:
            roc_curve_explained = st.expander(":interrobang: ROC Curve Explained")
            with roc_curve_explained:
                st.markdown(explained_dict["roc_curve_explained"])
                
            fig, ax = plt.subplots()
            fpr, tpr, thresholds = roc_curve(models[model]["y_test"], models[model]["predict_proba"][:,1]) 
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve 
            ax.plot(fpr, tpr, label = f"ROC curve (area = {roc_auc:0.2f})")
            ax.plot([0, 1], [0, 1], 'k--', label='No Skill')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
            st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["roc_auc"])

    # Model Explanations
    with h1n1_2:
        st.write("#### Variable Importance")
        
        feature_importance_explained = st.expander(":interrobang: Variable Importance Explained")
        with feature_importance_explained:
            st.markdown(explained_dict["feature_importance_explained"])

        st.write("Overall Variable Importance")
        st.plotly_chart(plots[model]["vi"])
        st.write("Protected Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Protected Characteristics"])
        st.write("Opinion-based Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Opinion Characteristics"])
        st.write("Bahaviour Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Behaviour Characteristics"])
        st.write("Health Variable Importance")
        st.plotly_chart(plots[model]["vi_Health Characteristics"])
        # st.write("Other Variable Importance")
        # st.plotly_chart(plots[model]["vi_Other Characteristics"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["vi"])

        st.write("#### Partial Dependence & Accumulated Local Dependence")
        
        partial_dependence_explained = st.expander(":interrobang: Partial Dependence & Accumulated Local Dependence Explained")
        with partial_dependence_explained:
            st.markdown(explained_dict["partial_dependence_explained"])

        st.plotly_chart(plots[model]["pdp_ale_num"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pd"])

        # st.plotly_chart(plots[model]["pdp_ale_cat"])
        # st.expander(":sparkles: Analysis").markdown('''Text''')

        st.write("#### Prediction Breakdown & Shapley Values")

        shapley_values_explained = st.expander(":interrobang: Shapley Values Explained")
        with shapley_values_explained:
            st.markdown(explained_dict["shapley_values_explained"])

        st.write("#### Most confident model prediction")
        st.write("#### Positive Class (i.e. Vaccinated = 1)")

        st.plotly_chart(plots[model]["bd_1"])

        st.plotly_chart(plots[model]["sh_1"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_1"])

        st.write("#### Negative Class (i.e. Not vaccinated = 0)")
        
        st.plotly_chart(plots[model]["bd_0"])

        st.plotly_chart(plots[model]["sh_0"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_0"])
        
        st.write("#### Most indecesive model prediction")
        
        st.plotly_chart(plots[model]["bd_05"])

        st.plotly_chart(plots[model]["sh_05"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_05"])

        # st.write("#### Ceteris Paribus Profiles for Most Confident Predictions")

        # ceteris_paribus_explained = st.expander(":interrobang: Ceteris Paribus Profiles Explained")
        # with ceteris_paribus_explained:
        #     st.markdown('''
        #                 Ceteris-paribus (CP) profiles show how a model’s prediction would change if the value of a single exploratory variable changed. In essence, a CP profile shows the dependence of the conditional expectation of the dependent variable (response) on the values of the particular explanatory variable.

        #                 [source](https://ema.drwhy.ai/ceterisParibus.html)
        #                 ''')

        # # st.plotly_chart(cp_1.plot(cp_0, show = False))
        # st.expander(":sparkles: Analysis").markdown('''Text''')

    # Model Fairness
    with h1n1_3:

        df_preds = pd.concat([
            models[model]["X_test"].reset_index(drop = True),
            models[model]["y_test"].reset_index(drop = True),
            pd.DataFrame(models[model]["predict"], columns = ["predict"]).reset_index(drop = True)],
            axis = 1)

        st.markdown('''
                    ### Simple Fairness Metrics
                    
                    Here we will investigate the fairness of the model regarding two key protected attributes: 
                    
                    :red-background[sex]
                    
                    :red-background[chronic_med_condition]
                    
                    A fair and unbiased model should not behave differently (i.e. generate different predictions) depending on whether someone is male of female, of if they have a chronical medical condition (i.e disability) or not.
                    ''')

        # Group Fairness
        st.markdown('''
                    #### Group Fairness
                    
                    :interrobang: Members of each group need to have the same probability of being assigned to the positively predicted class.
                    ''')
        
        h1n1_gf_1, h1n1_gf_2 = st.columns(2)
        
        gf_0 = ff.group_fairness(df_preds, "sex", "Male", "predict", 1)
        gf_1 = ff.group_fairness(df_preds, "sex", "Female", "predict", 1)
        h1n1_gf_1.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Sex | Male | {gf_0} |
                           | Sex | Female | {gf_1} |
                           
                           Difference: {np.abs(gf_0 - gf_1):.3f}
                            ''')
        h1n1_gf_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["gf"])

        # Group Fairness
        gf_0 = ff.group_fairness(df_preds, "chronic_med_condition", 0, "predict", 1)
        gf_1 = ff.group_fairness(df_preds, "chronic_med_condition", 1, "predict", 1)
        h1n1_gf_2.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Chronic Medical Condition | 0 (No) | {gf_0} |
                           | Chronic Medical Condition | 1 (Yes) | {gf_1} |
                           
                           Difference: {np.abs(gf_0 - gf_1):.3f}
                            ''')
        h1n1_gf_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["gf_1"])
        st.markdown("---")

        # Conditional Statistical Parity
        st.write('''
                 #### Conditional Statistical Parity
                    
                 :interrobang: Members of each group need to have the same probability of being assigned to the positive class under the same set of conditions.
                 ''')
        
        h1n1_csp_1, h1n1_csp_2 = st.columns(2)

        csp_0 = ff.conditional_statistical_parity(df_preds, "sex", "Male", "predict", 1, "doctor_recc_h1n1", 1)
        csp_1 = ff.conditional_statistical_parity(df_preds, "sex", "Female", "predict", 1, "doctor_recc_h1n1", 1)
        h1n1_csp_1.markdown(f'''
                            | Variable | Class | Value |
                            |---|---|---|
                            | Doctor Recommends H1N1 Vaccine + Sex | 1 (Yes) + Male | {csp_0} |
                            | Doctor Recommends H1N1 Vaccine + Sex | 1 (Yes) + Female | {csp_1} |
                            
                            Difference: {np.abs(csp_0 - csp_1):.3f}
                            ''')
        h1n1_csp_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["csp"])

        csp_0 = ff.conditional_statistical_parity(df_preds, "chronic_med_condition", 0, "predict", 1, "doctor_recc_h1n1", 1)
        csp_1 = ff.conditional_statistical_parity(df_preds, "chronic_med_condition", 1, "predict", 1, "doctor_recc_h1n1", 1)
        h1n1_csp_2.markdown(f'''
                            | Variable | Class | Value |
                            |---|---|---|
                            | Doctor Recommends H1N1 Vaccine + Chronic Medical Condition | 1 (Yes) + 0 (No) | {csp_0} |
                            | Doctor Recommends H1N1 Vaccine + Chronic Medical Condition | 1 (Yes) + 1 (Yes) | {csp_1} |
                            
                            Difference: {np.abs(csp_0 - csp_1):.3f}
                            ''')
        h1n1_csp_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["csp_1"])
        st.markdown("---")

        st.write('''
                 #### Predictive Parity
                    
                 :interrobang: Members of each group have the same Positive Predictive Value (PPV) — the probability of a subject with Positive Predicted Value to truly belong to the positive class.
                 ''')
        
        h1n1_pp_1, h1n1_pp_2 = st.columns(2)

        # Predictive Parity
        pp_0 = ff.predictive_parity(df_preds, "sex", "Male", "predict", model)
        pp_1 = ff.predictive_parity(df_preds, "sex", "Female", "predict", model)
        h1n1_pp_1.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Sex | Male | {pp_0:.3f} |
                           | Sex | Female | {pp_1:.3f} |
                           
                           Difference: {np.abs(pp_0 - pp_1):.3f}
                            ''')
        h1n1_pp_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pp"])

        pp_0 = ff.predictive_parity(df_preds, "chronic_med_condition", 0, "predict", model)
        pp_1 = ff.predictive_parity(df_preds, "chronic_med_condition", 1, "predict", model)
        h1n1_pp_2.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Chronic Medical Condition | 0 (No) | {pp_0:.3f} |
                           | Chronic Medical Condition | 1 (Yes) | {pp_1:.3f} |
                           
                           Difference: {np.abs(pp_0 - pp_1):.3f}
                            ''')
        h1n1_pp_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pp_1"])
        st.markdown("---")

        st.write(f'''
                 #### False Positive Error Rate Balance
                    
                 :interrobang: Members of each group have the same False Positive Rate (FPR) — the probability of a subject in the negative class to have a positive predicted value.
                 ''')
        
        h1n1_ffperb_1, h1n1_ffperb_2 = st.columns(2)
        # False Positive Error Rate Balance

        ffperb_0 = ff.fp_error_rate_balance(df_preds, "sex", "Male", "predict", model)
        ffperb_1 = ff.fp_error_rate_balance(df_preds, "sex", "Female", "predict", model)
        h1n1_ffperb_1.markdown(f'''
                               | Variable | Class | Value |
                               |---|---|---|
                               | Sex | Male | {ffperb_0:.3f} |
                               | Sex | Female | {ffperb_1:.3f} |
                               
                               Difference: {np.abs(ffperb_0 - ffperb_1):.3f}
                                ''')
        h1n1_ffperb_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["fperb"])

        ffperb_0 = ff.fp_error_rate_balance(df_preds, "chronic_med_condition", 0, "predict", model)
        ffperb_1 = ff.fp_error_rate_balance(df_preds, "chronic_med_condition", 1, "predict", model)
        h1n1_ffperb_2.markdown(f'''
                               | Variable | Class | Value |
                               |---|---|---|
                               | Chronic Medical Condition | 0 (No) | {ffperb_0:.3f} |
                               | Chronic Medical Condition | 1 (Yes) | {ffperb_1:.3f} |
                               
                               Difference: {np.abs(ffperb_0 - ffperb_1):.3f}
                                ''')
        h1n1_ffperb_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["fperb_1"])
        st.markdown("---")
        
        st.write("### Advanced Fairness Metrics")
        st.write('''
                 Now investigate the WEIRD population, which stands for:
                 
                 - **W** estern
                 
                 - **E** ducated
                 
                 - **I** ndustrialised 
                 
                 - **R** ich 
                 
                 - **D** emocracy
                 ''')

        protected, privileged = st.columns(2)
        protected.markdown('''
                           **Protected attributes**:
                           
                           :red-background[ethnicity: \[\"Black\", \"Other or Multiple\", \"Hispanic\"\]]
                            
                           :red-background[education: \[\"< 12 Years\", \"12 Years\", \"Some College\"\]]
                           
                           :red-background[income_poverty: \[\"Below Poverty\", \"<= $75,000, Above Poverty\"\]]
                           ''')
        
        privileged.markdown('''
                            **Privileged attribute**:
                            
                            :red-background[ethnicity: \"White\"]
                            
                            :red-background[education: \"College Graduate\"]
                            
                            :red-background[income_poverty: > \"$75,000\"]
                            ''')

        st.expander(":interrobang: Fairness Check explained").markdown(explained_dict["fairness_check_explained"])

        st.expander(":interrobang: Fairness Radar plot explained").markdown(explained_dict["fairness_radar_explained"])

        st.write("### Western")
        h1n1_ethnicity, h1n1_ethnicity_radar = st.columns(2)
        h1n1_ethnicity.plotly_chart(plots[model]["fobject_ethnicity"])
        h1n1_ethnicity.expander(":sparkles: Analysis").markdown(analysis_dict[model]["western"])
        h1n1_ethnicity_radar.plotly_chart(plots[model]["fobject_ethnicity_radar"])
        h1n1_ethnicity_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["western_radar"])
        st.markdown("---")

        st.write("### Educated")
        h1n1_education, h1n1_education_radar = st.columns(2)
        h1n1_education.plotly_chart(plots[model]["fobject_education"])
        h1n1_education.expander(":sparkles: Analysis").markdown(analysis_dict[model]["educated"])
        h1n1_education_radar.plotly_chart(plots[model]["fobject_education_radar"])
        h1n1_education_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["educated_radar"])
        st.markdown("---")
        
        st.write("### Rich")
        h1n1_rich, h1n1_rich_radar = st.columns(2)
        h1n1_rich.plotly_chart(plots[model]["fobject_income_poverty"])
        h1n1_rich.expander(":sparkles: Analysis").markdown(analysis_dict[model]["rich"])
        h1n1_rich_radar.plotly_chart(plots[model]["fobject_income_poverty_radar"])
        h1n1_rich_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["rich_radar"])
        
model = "seasonal_vaccine"

with seasonal_vaccine_tab:
    seasonal_1, seasonal_2, seasonal_3 = st.tabs([":chart_with_upwards_trend: Model Performance", ":bulb: Model Explanations", ":busts_in_silhouette: Model Fairness"])
    with seasonal_1:

        seasonal_p_dist_1, seasonal_class_report_1 = st.columns([2, 1])
        seasonal_p_dist_2, seasonal_class_report_2 = st.columns([2, 1], vertical_alignment = "center")
        seasonal_p_dist_3, seasonal_class_report_3 = st.columns([2, 1])

        fig = px.histogram(models[model]["predict_proba"][:,1], nbins = 10, title = f"MLP Classifier: Distribution of Predicted Values for {model}")
        fig.update_traces(marker_line_width = 1, marker_line_color = "black")
        fig.update_layout(xaxis_title = "Probability", yaxis_title = "Count", showlegend = False)
        seasonal_p_dist_1.expander(":interrobang: Distribution of Predicted Values Plot Explained").markdown(explained_dict["probability_distribution_explained"])
        seasonal_p_dist_2.plotly_chart(fig)
        seasonal_p_dist_3.expander(":sparkles: Analysis").markdown(analysis_dict[model]["prob_dist"])

        seasonal_class_report_1.expander(":interrobang: Classification Report Explained").markdown(explained_dict["classification_report_explained"])
        seasonal_class_report_2.dataframe(pd.DataFrame(classification_report(models[model]["y_test"], models[model]["predict"], output_dict = True)).T)
        seasonal_class_report_3.expander(":sparkles: Analysis").markdown(analysis_dict[model]["class_report"])
        
        st.markdown("---")
        seasonal_cf, seasonal_roc = st.columns(2)
        with seasonal_cf:
            confusion_matrix_explained = st.expander(":interrobang: Confusion Matrix Explained")
            with confusion_matrix_explained:
                st.markdown(explained_dict["confusion_matrix_explained"])

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(models[model]["y_test"], models[model]["predict"])).plot(ax = ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["cf"])

        with seasonal_roc:
            roc_curve_explained = st.expander(":interrobang: ROC Curve Explained")
            with roc_curve_explained:
                st.markdown(explained_dict["roc_curve_explained"])
                
            fig, ax = plt.subplots()
            fpr, tpr, thresholds = roc_curve(models[model]["y_test"], models[model]["predict_proba"][:,1]) 
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve 
            ax.plot(fpr, tpr, label = f"ROC curve (area = {roc_auc:0.2f})")
            ax.plot([0, 1], [0, 1], 'k--', label='No Skill')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
            st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["roc_auc"])

    # Model Explanations
    with seasonal_2:
        st.write("#### Variable Importance")
        
        feature_importance_explained = st.expander(":interrobang: Variable Importance Explained")
        with feature_importance_explained:
            st.markdown(explained_dict["feature_importance_explained"])

        st.write("Overall Variable Importance")
        st.plotly_chart(plots[model]["vi"])
        st.write("Protected Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Protected Characteristics"])
        st.write("Opinion-based Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Opinion Characteristics"])
        st.write("Bahaviour Characteristics Variable Importance")
        st.plotly_chart(plots[model]["vi_Behaviour Characteristics"])
        st.write("Health Variable Importance")
        st.plotly_chart(plots[model]["vi_Health Characteristics"])
        # st.write("Other Variable Importance")
        # st.plotly_chart(plots[model]["vi_Other Characteristics"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["vi"])

        st.write("#### Partial Dependence & Accumulated Local Dependence")
        
        partial_dependence_explained = st.expander(":interrobang: Partial Dependence & Accumulated Local Dependence Explained")
        with partial_dependence_explained:
            st.markdown(explained_dict["partial_dependence_explained"])

        st.plotly_chart(plots[model]["pdp_ale_num"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pd"])

        # st.plotly_chart(plots[model]["pdp_ale_cat"])
        # st.expander(":sparkles: Analysis").markdown('''Text''')

        st.write("#### Prediction Breakdown & Shapley Values")

        shapley_values_explained = st.expander(":interrobang: Shapley Values Explained")
        with shapley_values_explained:
            st.markdown(explained_dict["shapley_values_explained"])

        st.write("#### Most confident model prediction")
        st.write("#### Positive Class (i.e. Vaccinated = 1)")

        st.plotly_chart(plots[model]["bd_1"])

        st.plotly_chart(plots[model]["sh_1"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_1"])

        st.write("#### Negative Class (i.e. Not vaccinated = 0)")
        
        st.plotly_chart(plots[model]["bd_0"])

        st.plotly_chart(plots[model]["sh_0"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_0"])
        
        st.write("#### Most indecesive model prediction")
        
        st.plotly_chart(plots[model]["bd_05"])

        st.plotly_chart(plots[model]["sh_05"])
        st.expander(":sparkles: Analysis").markdown(analysis_dict[model]["shap_vals_05"])

        # st.write("#### Ceteris Paribus Profiles for Most Confident Predictions")

        # ceteris_paribus_explained = st.expander(":interrobang: Ceteris Paribus Profiles Explained")
        # with ceteris_paribus_explained:
        #     st.markdown('''
        #                 Ceteris-paribus (CP) profiles show how a model’s prediction would change if the value of a single exploratory variable changed. In essence, a CP profile shows the dependence of the conditional expectation of the dependent variable (response) on the values of the particular explanatory variable.

        #                 [source](https://ema.drwhy.ai/ceterisParibus.html)
        #                 ''')

        # # st.plotly_chart(cp_1.plot(cp_0, show = False))
        # st.expander(":sparkles: Analysis").markdown('''Text''')

    # Model Fairness
    with seasonal_3:

        df_preds = pd.concat([
            models[model]["X_test"].reset_index(drop = True),
            models[model]["y_test"].reset_index(drop = True),
            pd.DataFrame(models[model]["predict"], columns = ["predict"]).reset_index(drop = True)],
            axis = 1)

        st.markdown('''
                    ### Simple Fairness Metrics
                    
                    Here we will investigate the fairness of the model regarding two key protected attributes: 
                    
                    :red-background[sex]
                    
                    :red-background[chronic_med_condition]
                    
                    A fair and unbiased model should not behave differently (i.e. generate different predictions) depending on whether someone is male of female, of if they have a chronical medical condition (i.e disability) or not.
                    ''')

        # Group Fairness
        st.markdown('''
                    #### Group Fairness
                    
                    :interrobang: Members of each group need to have the same probability of being assigned to the positively predicted class.
                    ''')
        
        seas_gf_1, seas_gf_2 = st.columns(2)
        
        gf_0 = ff.group_fairness(df_preds, "sex", "Male", "predict", 1)
        gf_1 = ff.group_fairness(df_preds, "sex", "Female", "predict", 1)
        seas_gf_1.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Sex | Male | {gf_0} |
                           | Sex | Female | {gf_1} |
                           
                           Difference: {np.abs(gf_0 - gf_1):.3f}
                            ''')
        seas_gf_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["gf"])

        # Group Fairness
        gf_0 = ff.group_fairness(df_preds, "chronic_med_condition", 0, "predict", 1)
        gf_1 = ff.group_fairness(df_preds, "chronic_med_condition", 1, "predict", 1)
        seas_gf_2.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Chronic Medical Condition | 0 (No) | {gf_0} |
                           | Chronic Medical Condition | 1 (Yes) | {gf_1} |
                           
                           Difference: {np.abs(gf_0 - gf_1):.3f}
                            ''')
        seas_gf_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["gf_1"])
        st.markdown("---")

        # Conditional Statistical Parity
        st.write('''
                 #### Conditional Statistical Parity
                    
                 :interrobang: Members of each group need to have the same probability of being assigned to the positive class under the same set of conditions.
                 ''')
        
        seas_csp_1, seas_csp_2 = st.columns(2)

        csp_0 = ff.conditional_statistical_parity(df_preds, "sex", "Male", "predict", 1, "doctor_recc_h1n1", 1)
        csp_1 = ff.conditional_statistical_parity(df_preds, "sex", "Female", "predict", 1, "doctor_recc_h1n1", 1)
        seas_csp_1.markdown(f'''
                            | Variable | Class | Value |
                            |---|---|---|
                            | Doctor Recommends H1N1 Vaccine + Sex | 1 (Yes) + Male | {csp_0} |
                            | Doctor Recommends H1N1 Vaccine + Sex | 1 (Yes) + Female | {csp_1} |
                            
                            Difference: {np.abs(csp_0 - csp_1):.3f}
                            ''')
        seas_csp_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["csp"])

        csp_0 = ff.conditional_statistical_parity(df_preds, "chronic_med_condition", 0, "predict", 1, "doctor_recc_h1n1", 1)
        csp_1 = ff.conditional_statistical_parity(df_preds, "chronic_med_condition", 1, "predict", 1, "doctor_recc_h1n1", 1)
        seas_csp_2.markdown(f'''
                            | Variable | Class | Value |
                            |---|---|---|
                            | Doctor Recommends H1N1 Vaccine + Chronic Medical Condition | 1 (Yes) + 0 (No) | {csp_0} |
                            | Doctor Recommends H1N1 Vaccine + Chronic Medical Condition | 1 (Yes) + 1 (Yes) | {csp_1} |
                            
                            Difference: {np.abs(csp_0 - csp_1):.3f}
                            ''')
        seas_csp_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["csp_1"])
        st.markdown("---")

        st.write('''
                 #### Predictive Parity
                    
                 :interrobang: Members of each group have the same Positive Predictive Value (PPV) — the probability of a subject with Positive Predicted Value to truly belong to the positive class.
                 ''')
        
        seas_pp_1, seas_pp_2 = st.columns(2)

        # Predictive Parity
        pp_0 = ff.predictive_parity(df_preds, "sex", "Male", "predict", model)
        pp_1 = ff.predictive_parity(df_preds, "sex", "Female", "predict", model)
        seas_pp_1.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Sex | Male | {pp_0:.3f} |
                           | Sex | Female | {pp_1:.3f} |
                           
                           Difference: {np.abs(pp_0 - pp_1):.3f}
                            ''')
        seas_pp_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pp"])

        pp_0 = ff.predictive_parity(df_preds, "chronic_med_condition", 0, "predict", model)
        pp_1 = ff.predictive_parity(df_preds, "chronic_med_condition", 1, "predict", model)
        seas_pp_2.markdown(f'''
                           | Variable | Class | Value |
                           |---|---|---|
                           | Chronic Medical Condition | 0 (No) | {pp_0:.3f} |
                           | Chronic Medical Condition | 1 (Yes) | {pp_1:.3f} |
                           
                           Difference: {np.abs(pp_0 - pp_1):.3f}
                            ''')
        seas_pp_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["pp_1"])
        st.markdown("---")

        st.write(f'''
                 #### False Positive Error Rate Balance
                    
                 :interrobang: Members of each group have the same False Positive Rate (FPR) — the probability of a subject in the negative class to have a positive predicted value.
                 ''')
        
        seas_ffperb_1, seas_ffperb_2 = st.columns(2)
        # False Positive Error Rate Balance

        ffperb_0 = ff.fp_error_rate_balance(df_preds, "sex", "Male", "predict", model)
        ffperb_1 = ff.fp_error_rate_balance(df_preds, "sex", "Female", "predict", model)
        seas_ffperb_1.markdown(f'''
                               | Variable | Class | Value |
                               |---|---|---|
                               | Sex | Male | {ffperb_0:.3f} |
                               | Sex | Female | {ffperb_1:.3f} |
                               
                               Difference: {np.abs(ffperb_0 - ffperb_1):.3f}
                                ''')
        seas_ffperb_1.expander(":sparkles: Analysis").markdown(analysis_dict[model]["fperb"])

        ffperb_0 = ff.fp_error_rate_balance(df_preds, "chronic_med_condition", 0, "predict", model)
        ffperb_1 = ff.fp_error_rate_balance(df_preds, "chronic_med_condition", 1, "predict", model)
        seas_ffperb_2.markdown(f'''
                               | Variable | Class | Value |
                               |---|---|---|
                               | Chronic Medical Condition | 0 (No) | {ffperb_0:.3f} |
                               | Chronic Medical Condition | 1 (Yes) | {ffperb_1:.3f} |
                               
                               Difference: {np.abs(ffperb_0 - ffperb_1):.3f}
                                ''')
        seas_ffperb_2.expander(":sparkles: Analysis").markdown(analysis_dict[model]["fperb_1"])
        st.markdown("---")
        
        st.write("### Advanced Fairness Metrics")
        st.write('''
                 Now investigate the WEIRD population, which stands for:
                 
                 - **W** estern
                 
                 - **E** ducated
                 
                 - **I** ndustrialised 
                 
                 - **R** ich 
                 
                 - **D** emocracy
                 ''')
        
        protected, privileged = st.columns(2)
        protected.markdown('''
                           **Protected attributes**:
                           
                           :red-background[ethnicity: \[\"Black\", \"Other or Multiple\", \"Hispanic\"\]]
                            
                           :red-background[education: \[\"< 12 Years\", \"12 Years\", \"Some College\"\]]
                           
                           :red-background[income_poverty: \[\"Below Poverty\", \"<= $75,000, Above Poverty\"\]]
                           ''')
        
        privileged.markdown('''
                            **Privileged attribute**:
                            
                            :red-background[ethnicity: \"White\"]
                            
                            :red-background[education: \"College Graduate\"]
                            
                            :red-background[income_poverty: > \"$75,000\"]
                    ''')
        
        st.expander(":interrobang: Fairness Check explained").markdown(explained_dict["fairness_check_explained"])

        st.expander(":interrobang: Fairness Radar plot explained").markdown(explained_dict["fairness_radar_explained"])
        
        st.write("### Western")
        seas_ethnicity, seas_ethnicity_radar = st.columns(2)
        seas_ethnicity.plotly_chart(plots[model]["fobject_ethnicity"])
        seas_ethnicity.expander(":sparkles: Analysis").markdown(analysis_dict[model]["western"])
        seas_ethnicity_radar.plotly_chart(plots[model]["fobject_ethnicity_radar"])
        seas_ethnicity_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["western_radar"])
        st.markdown("---")

        st.write("### Educated")
        seas_education, seas_education_radar = st.columns(2)
        seas_education.plotly_chart(plots[model]["fobject_education"])
        seas_education.expander(":sparkles: Analysis").markdown(analysis_dict[model]["educated"])
        seas_education_radar.plotly_chart(plots[model]["fobject_education_radar"])
        seas_education_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["educated_radar"])
        st.markdown("---")
        
        st.write("### Rich")
        seas_rich, seas_rich_radar = st.columns(2)
        seas_rich.plotly_chart(plots[model]["fobject_income_poverty"])
        seas_rich.expander(":sparkles: Analysis").markdown(analysis_dict[model]["rich"])
        seas_rich_radar.plotly_chart(plots[model]["fobject_income_poverty_radar"])
        seas_rich_radar.expander(":sparkles: Analysis").markdown(analysis_dict[model]["rich_radar"])