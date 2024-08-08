class msg():
    FORMULA_TOKEN = 'FORMULA'
    # Headers
    MAIN_TITLE = '🚑 Heart Disease Factors Data App'
    FILTER_TITLE = '🔬 Filters'
    FEATURE_TITLE = '📊 Features Information'
    DATA_DESCRIPTION_TITLE = '📜 Dataset Description'
    FULL_DATASET_TITLE = '### Cleveland Heart Disease Dataset (full)'
    FILTERED_DATASET_TITLE = '### Cleveland Heart Disease Dataset (filtered)'
    FEATURE_DISTRIBUTION_TITLE = '### Individual Feature Distributions (filtered)'
    FEATURE_CORRELATION_TITLE = '### Pairwise Feature Distributions (filtered)'
    MODEL_DUMMY_TITLE = '### Dummy Model'
    PERFOMANCE_DUMMY_TITLE = '#### 🐌 Dummy Model Perfomance'
    MODEL_LOGREG_TITLE = 'Simple Logistic Regression'
    MODEL_FOREST_TITLE = 'Random Forest'
    MODEL_BOOST_TITLE = '### Gradient Boosting'
    FAIRNESS_LOGREG_TITLE = '### Fairness Metrics for Logistic Regression'
    FAIRNESS_LOGREG_BALANCED_TITLE = '### Fairness Metrics for Logistic Regression (Balanced)'
    FAIRNESS_RF_TITLE = '### Fairness Metrics for Random Forest'
    FAIRNESS_RF_BALANCED_TITLE = '### Fairness Metrics for Random Forest (Balanced)'
    FAIRNESS_DISCUSS_TITLE = '### Discussion of Fairness Metrics'
    FAIRNESS_BALANCED_DISCUSS_TITLE = '### Discussion of Fairness Metrics (Balanced)'
    XAI_LOGREG_TITLE = '### Explaining Logistic Regression'
    XAI_BOOST_TITLE = '### Explaining Gradient Boosting'
    XAI_BOOST_INDIVIDUAL_TITLE = '#### 🤒❓ Explaining Individual Samples'
    XAI_BOOST_GLOBAL_TITLE = '#### 👨‍👩‍👧‍👦❓ Global Explanations'
    MODELS_TITLE = "Model Explanation"
    FEATURE_IMPORTANCE = "Feature Importance (Coefficients)"
    FEATURE_COEFFICIENTS = "Feature Coefficients"

    # Text snippets
    SELECT_FEATURE_MSG = 'Select the feature you would like to know more about!'
    FULL_DATASET_MSG = 'Here you can see the entire dataset, unaffected by filters.'
    FITERED_DATASET_MSG = 'Adjust the filters on the sidebar to see how the dataset changes. This dataset also has no missing values, \
                           since rows with them have been removed.'
    FEATURE_DISTRIBUTION_MSG = 'Select the feature to see their distribution. You can also interact with the filters to get more specific distributions.'
    FEATURE_CORRELATION_MSG = 'Select a pair of features to see their mutual distribution. Filters affect this too!'
    LOGREG_TRAIN_MSG = '👇 Click the button to train a logistic regression model'
    FOREST_TRAIN_MSG = '👇 Click the button to train a random forest model'
    BOOST_TRAIN_MSG = '👇 Click the button to train a gradient boosting model'
    FOREST_HYPERPARAMS_MSG = '🔧 You can adjust the model\'s hyperparameters with the options below'
    FAIRNESS_LOGREG_MSG = 'First off, let\'s see the fairness metrics for Logistic Regression model.'
    FAIRNESS_RF_MSG = 'Now, let\'s see the fairness metrics for the Random Forest model.'
    FAIRNESS_MOD_LR_MSG = 'To mitigate differences in metrics, we can try to remove the *Sex* feature from the training dataset of Logistic Regression Model. Here are the results.'
    FAIRNESS_MOD_RF_MSG = 'Similar to Logistic Regression, In order to mitigate differences in metrics, we can try to remove the *Sex* feature from the training dataset of Random Forest Model. Here are the results.'

    # Longtexts
    APP_DESCRIPTION = 'This app provides an outlook on Cleveland Heart Disease dataset. \
                       You can examine surface-level data analysis, as well as predicitons of \
                       simple machine learning models and their fairness.\n \n For more information:\n \
                       https://archive.ics.uci.edu/dataset/45/heart+disease'
    DATASET_DESCRIPTION = ['*Introductory paper*: https://www.semanticscholar.org/paper/International-application-of-a-new-probability-for-Detrano-Jánosi/a7d714f8f87bfc41351eb5ae1e5472f0ebbe0574',
                           '*Created by*: Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano',
                           '*Created in*: 1989',
                           '*Purpose of creation*: validation of a newly proposed probbility model',
                           '*Data collection method*: medical examination']
    MODELS_DESCRIPTION = ['On this tab you can explore the work of diffrent classification models which predict \
                          whether a patient is likely to have a heart disease', '❗<b>IMPORTANT</b>❗: for simplicity, we combine labels 1 to 4 \
                          together. This allows for the simpler binary \
                          classification and more interpretable metrics.', '❗<b>IMPORTANT</b>❗: data will <b>not</b> \
                          be filtered for training the models.'
                          ]
    MODELS_WARNING = f"""
        <div style="border: 2px solid #FF0000; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <p>{MODELS_DESCRIPTION[1]}</p>
            <p>{MODELS_DESCRIPTION[2]}</p>
        </div>
        """
    MODEL_LOGREG_DESCRIPTION = 'Logistic regression is a simple classifier and one of the most iconic and used machine learning algorithms. \
                               It is a statistical model used for binary classification problems, where the goal is to predict the probability \
                               of one of two possible outcomes. It models the relationship between a dependent binary variable and one or more \
                              independent variables using the logistic function. The logistic function maps any real-valued number into a value \
                              between 0 and 1, which can be interpreted as a probability. Logistic Regression estimates the coefficients of the model \
                              using maximum likelihood estimation, providing a straightforward and interpretable way to understand the influence of each predictor on the outcome.\
                              Let\'s see how it tries to build predictions for our dataset.\
                              **Click here for more information about Logistic Regression:** https://www.ibm.com/topics/logistic-regression'
    MODEL_FOREST_DESCRIPTION = 'Random Forest is an ensemble learning method primarily used for classification and regression tasks. It constructs multiple \
                      decision trees during training and merges them to improve the predictive accuracy and control overfitting. Each tree is built from a random subset \
                      of the training data and a random subset of features, which introduces variability and robustness. The final prediction is made by \
                      aggregating the predictions of all individual trees, typically using majority voting for classification or averaging for regression.\
                      **Click the link for more information about Random Forest:** https://www.ibm.com/topics/random-forest'
    FAIRNESS_DESCRIPTION = ['On this tab you can explore different metrics associated with gender fairness, \
                            and see how fairly our models fare.']
    FAIRNESS_DISCUSSION = 'In reality, it makes little to no sense to assess the Group Fairness and Conditional Statistical Fairness metrics, since the frequency of heart disease \
                           is inherently different in men and women. \n On the other hand, metrics such as Predictive Parity or False Positive Error Rate Balance \
                           are useful, since they show us if the model makes the same amount of mistakes on men and women.'
    FAIRNESS_DISCUSSION_BALANCED= """
      After balancing the dataset based on the 'sex' feature, we can observe the following changes in fairness metrics:

    1. Group Fairness: The probabilities of being assigned to the positive class (heart disease prediction) for males did not change much for both male and female groups for both the models. :blue-background[No significant improvement in group fairness.]

    2. Conditional Statistical Fairness: The probabilities for both sexes to get heart disease, given the condition of having diabetes (fbs = 1), are more unbalanced now for Logistic regression and Random forest model. :blue-background[Slight decline in conditional statistical fairness.]

    3. Predictive Parity: The Positive Predictive Values (PPV) reduced slightly for both females and males for Logistic Regression and improved for Random forest models for males. :blue-background[Improved for Random Forest but worsened for Logistic Regression.]

    4. False Positive Error Rate Balance: The False Positive Rates (FPR) increased for both the sexes for Logistic Regression and remained unchanged for Random forest model for both the sexes. :blue-background[Improved for Random Forest but worsened for Logistic Regression.]

    :blue-background[This concludes that oversampling for balancing the two groups males and females in the input dataset did not lead to much improvement in group fairness and conditional statistical fairness for both the models but positive predictive values and false positive error rate balance improved for Random Forest model and worsened for Logistic Regression model]
    """
    
    XAI_DESCRIPTION = 'Perhaps you are wondering "How do the models make decisions?" There are a lot of ways of gaining deeper \
                       understanding: sometimes models are inherently explainable, and sometimes we need advanced explanation techniques. \
                       Let\'s look at some of the models from the previous tab and try to get insight for their behaviour.'
    XAI_LOGREG_DESCRIPTION = ['Logistic regression can be explained fairly easily. Let\'s look at the decision function:',
                              FORMULA_TOKEN + r'f(\textbf{x}) = \frac{1}{1 + \exp{(- \textbf{w}^\text{T}\textbf{x} + b)}},',
                              'where *f* outputs the probabillity of the person being ill, **x** is the feature vector, \
                               and **w** is the weight vector which defines our logistic regression model. We can show that',
                               FORMULA_TOKEN + r'\nabla_\textbf{x} f(\textbf{x}) = f(\textbf{x}) (1 - f(\textbf{x})) \textbf{w} \sim  \textbf{w}.',
                               'We can see that for each given input the small individual perturbations of the features lead to \
                               changes of the decision function proportional to **w**. Since all features are normalized in preprocessing, \
                               weights can tell us about the relevance of all features for model\'s decision-making. Here is the plot:'
                             ]
    XAI_LOGREG_DISCUSSION = 'Weights with higher absolute value correspond to more relevant features; features with positive weights \
                             shift the model\'s decision towards the "ill" label, features with negative weights \
                             shift the model\'s decision towards the "healthy" label.'
    XAI_RF_DESCRIPTION = 'Random Forest is a black-box model, meaning that it doesn\'t have a weight vector or some other way \
                             of being interpretable by default. We will use the SHAP values to gain some insight. SHAP values, in essence, \
                             tell us how a prediction would change if we were to "hide" the value of a certain feature from the model. \
                             Let\'s see them in action now!'
    XAI_BOOST_INDIVIDUAL_DISCUSSION = 'On the plot you can see the features which contributed the most to the prediction result. The red features \
                                       contribute to lableling the person as ill, while the blue features contribute to labeling the person as \
                                       healthy. Among red features you can often see things like high BMI, low health self-assessment or \
                                       high blood pressure. Among blue features you can often see things like normal cholesterol levels, \
                                       healthy BMI or high health self-assessment.'
    SHAP_VALUES_DISCUSSION = """
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;margin-bottom: 20px">
                <h3>Shap Values</h3>
                <p>SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance. 
                They represent the impact each feature has on the model's output, averaged across all predictions. 
                The Mean Absolute SHAP Value (MEAN_ABS_SHAP) indicates the average magnitude of a feature's effect on the predictions, 
                regardless of the direction (positive or negative).</p>
                <p>The chart above displays the feature importances as determined by the Random Forest model. 
                Higher SHAP values signify features that have a more significant impact on the model's output.</p>
                <ul>
                    <li><strong>Feature:</strong> The name of the feature from the dataset.</li>
                    <li><strong>Mean Absolute SHAP Value:</strong> The average magnitude of the SHAP values for the feature, indicating its overall importance.</li>
                </ul>
            </div>
            """
    SHAP_VALUES_INDIVIDUAL_DISCUSSION = "Now let's look at individual SHAP values"
    XAI_RF_GLOBAL_DISCUSSION = 'Here we can see the average relevance for different values of different features. If the feature is located low, \
                                   all its values are located near the 0.0 line, which means that these features are almost irrelevant. Higher features \
                                   have higher importance. Let\'s look at a few examples.'
    XAI_BOOST_CA_DISCUSSION = '**ca**: We can see that the highest ca (Number of major vessels colored by flourosopy) values often have maximal relevance and shift the model to label the person \
                                as ill. At the same time, the lowest values encourage the model to label \
                                the person as healthy.'
    XAI_BOOST_FBS_DISCUSSION = '**fbs**: *fbs* is a binary variable. We can see that if a person doesn\'t have high blood sugar, \
                                   the feature is not at all relevant for the model. If the person does have a high blood sugar, \
                                   the feature may be somewhat relevant for the prediction, and shifts the decision towards an \'ill\' label.'
    
    # Feature exploration dictionaries
    FEATURES_DESC = {'age': 'The patient\'s age in years',
                     'sex': 'Biological sex',
                     'cp': 'Chest pain type',
                     'trestbps': 'Resting blood pressure (on admission to the hospital)',
                     'chol': 'Serum cholestoral in **mg/dl**',
                     'fbs': 'Is fasting blood sugar higher than 120 **mg/dl**?',
                     'restecg': 'Resting electrocardiographic results',
                     'thalach': 'Maximum heart rate achieved',
                     'exang': 'Exercise induced angina',
                     'oldpeak': 'ST depression induced by exercise relative to rest. For more details see \
                                https://en.wikipedia.org/wiki/ST_depression',
                     'slope': 'The slope of the peak exercise ST segment. For more details see \
                               https://en.wikipedia.org/wiki/ST_segment',
                     'ca': 'Number of major vessels colored by flourosopy',
                     'thal': 'A "thallium defect" refers to an area of the heart muscle that shows reduced \
                        uptake of thallium, indicating reduced blood flow to that region. There are different types of defects \
                        that can be identified',
                     'num': 'Presence of heart disease'}
    FEATURES_VALS = {'age': 'Ordered integer value in range $[29, 77]$',
                     'sex': '0 = female, 1 = male',
                     'cp': '1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic. For details visit \
                            https://www.textbookofcardiology.org/wiki/Chest_Pain_/_Angina_Pectoris',
                      'trestbps': 'Ordered integer value in range $[94, 200]$',
                      'chol': 'Ordered integer value in range $[126, 564]$',
                      'fbs': '0 = no, 1 = yes',
                      'restecg': '0 = normal, 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or \
                       depression of > 0.05 mV), 2 = showing probable or definite left ventricular hypertrophy by Estes\' criteria',
                      'thalach': 'Ordered integer value in range $[71, 202]$',
                      'exang': '0 = absent, 1 = present',
                      'oldpeak': 'Continuous value in range $[0.0, 6.2]$',
                      'slope': '1 = upsloping, 2 = flat, 3 = downsloping',
                      'ca': 'Amount of vessels (0-3)',
                      'thal': '3 = normal, 6 = fixed defect, 7 = reversable defect',
                      'num': '0 - healthy, 1-4 - ill'}
    
    # Error messages
    EMPTY_FILTER_ERROR = 'Please select a non-empty filtered set 😭😭😭'
    MISSING_FEATURE_ERROR = '😨😳😖 Whoops! Unexpected key error - please contact the developer'
    NO_LOGREG_ERROR = '🤷‍♂️🤷‍♀️ No trained logistic regression detected - please visit the previous tab to train the model!'
    NO_BOOST_ERROR = '🤷‍♂️🤷‍♀️ No trained gradient boosting detected - please visit the previous tab to train the model!'
    TRIVIAL_BOOST_ERROR = '❗ **WARNING** ❗ A trivial classifier was learned - evaluation impossible.'
    OVERFIT_WARNING = '🕵️‍♂️ *Automated Overfit Detection*: final losses are disparate for train and test data. \
                        Using a simpler model is advised.'

    # Miscellaneous
    SEPARATOR = '_' * 15

    # Fairnes metrics
    FF_METRIC_1 = "Group Fairness"
    FF_METRIC_2 = "Conditional Statistical Fairness (conditional on diabetes feature)"
    FF_METRIC_3 = "Predictive Parity"
    FF_METRIC_4 = "False Positive Error Rate Balance"

    # Fairness Tab Introduction
    FF_METRICS_INTRODUCTION="""
    #### Group Fairness

    This metric checks if members of each group have the same probability of being assigned to the positive class.
    In our case we will define the positive class as being predicted having heart disease. 

    For example, if we investigate the group 'Sex' then coinciding Group Fairness metrics would imply that
    :blue-background[both groups have the same probability to receive an 'ill' prediction.] Mathematically this is stated as follows:
    
    P(Prediction = 1 | Sex = female) == P(Prediction = 1 | Sex = male)  

    If we calculate this probability for both groups we can then compare them and check if Group Fairness is present or not.

    #### Conditional Statistical Fairness

    This metric checks if members of each group have the same probability of being assigned to the positive class *under the same set of conditions*.

    For example, equal Conditional Statistical Fairness with respect to the 'fbs' feature would imply
    
    P(HeartDiseasePrediction = 1 | Sex = female, fbs = 1) == P(HeartDiseasePrediction = 1 | Sex = male, fbs = 1)

    This formula implies that :blue-background[both diabetic men and women have the same probability to receive an 'ill' prediction.]

    Group Fairness as well as the Conditional Statistical Fairness both only consider the predictions of the machine learning model. 
    The two upcoming metrics do additionally include the ground truth of each prediction.

    #### Predictive Parity

    Positive Predictive Value (PPV) is the probability of a subject who is predicted as 'ill' to truly have heart disease.

    :blue-background[The PPV is calculated as:  True Positives / (True Positives + False Positives)]

    A high PPV indicates that we can be sure that a positive prediction is true. Ideally this value is 1, then we can be certain that any positive prediction is true.
    
    :blue-background[Predcitive Parity checks whether men and women get the same values of PPV from model's predictions.]

    #### False Positive Error Rate Balance

    False Positive Rate (FPR) is the probability of a subject who is predicted as 'ill' to actually be healthy.

    :blue-background[The FPR is calculated as:  False Positives / (False Positives + True Negatives)]

    :blue-background[False Positive Error Rate Balance checks whether men and women get the same values of FPR from model's predictions.]
    """