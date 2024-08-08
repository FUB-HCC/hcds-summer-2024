import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import pickle

plotly.offline.init_notebook_mode()

with open("df.pkl", "rb") as file:
    # Deserialize and load the object from the file
    df = pickle.load(file)

st.set_page_config(page_title = "Data", layout = "wide")

st.title(":bar_chart: Data")

data_exploration, data_documentation = st.tabs([":mag_right: Data Exploration", ":page_facing_up: Documentation"])

with data_exploration:

    data_dictionary = {
        "h1n1_vaccine": {
            "description": "Whether respondent received H1N1 flu vaccine.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "seasonal_vaccine": {  
            "description": "Whether respondent received seasonal flu vaccine.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "respondent_id": {
            "description": "Unique and random identifier.",
            "mappings": {
                None
            }
        },
        "h1n1_concern": {
            "description": "Level of concern about the H1N1 flu.",
            "mappings": {
                0: "Not at all concerned",
                1: "Not very concerned",
                2: "Somewhat concerned",
                3: "Very concerned",
                999: "Missing"
            }
        },
        "h1n1_knowledge": {
            "description": "Level of knowledge about H1N1 flu.",
            "mappings": {
                0: "No knowledge",
                1: "A little knowledge",
                2: "A lot of knowledge",
                999: "Missing"
            }
        },
        "behavioral_antiviral_meds": {
            "description": "Has taken antiviral medications.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_avoidance": {
            "description": "Has avoided close contact with others with flu-like symptoms.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_face_mask": {
            "description": "Has bought a face mask.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_wash_hands": {
            "description": "Has frequently washed hands or used hand sanitizer.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_large_gatherings": {
            "description": "Has reduced time at large gatherings.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_outside_home": {
            "description": "Has reduced contact with people outside of own household.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "behavioral_touch_face": {
            "description": "Has avoided touching eyes, nose, or mouth.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "doctor_recc_h1n1": {
            "description": "H1N1 flu vaccine was recommended by doctor.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "doctor_recc_seasonal": {
            "description": "Seasonal flu vaccine was recommended by doctor.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "chronic_med_condition": {
            "description": "Has any of the listed chronic medical conditions.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "child_under_6_months": {
            "description": "Has regular close contact with a child under the age of six months.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "health_worker": {
            "description": "Is a healthcare worker.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "health_insurance": {
            "description": "Has health insurance.",
            "mappings": {
                0: "No",
                1: "Yes",
                999: "Missing"
            }
        },
        "opinion_h1n1_vacc_effective": {
            "description": "Respondent's opinion about H1N1 vaccine effectiveness.",
            "mappings": {
                1: "Not at all effective",
                2: "Not very effective",
                3: "Don't know",
                4: "Somewhat effective",
                5: "Very effective",
                999: "Missing"
            }
        },
        "opinion_h1n1_risk": {
            "description": "Respondent's opinion about risk of getting sick with H1N1 flu without vaccine.",
            "mappings": {
                1: "Very Low",
                2: "Somewhat low",
                3: "Don't know",
                4: "Somewhat high",
                5: "Very high",
                999: "Missing"
            }
        },
        "opinion_h1n1_sick_from_vacc": {
            "description": "Respondent's worry of getting sick from taking H1N1 vaccine.",
            "mappings": {
                1: "Not at all worried",
                2: "Not very worried",
                3: "Don't know",
                4: "Somewhat worried",
                5: "Very worried",
                999: "Missing"
            }
        },
        "opinion_seas_vacc_effective": {
            "description": "Respondent's opinion about seasonal flu vaccine effectiveness.",
            "mappings": {
                1: "Not at all effective",
                2: "Not very effective",
                3: "Don't know",
                4: "Somewhat effective",
                5: "Very effective",
                999: "Missing"
            }
        },
        "opinion_seas_risk": {
            "description": "Respondent's opinion about risk of getting sick with seasonal flu without vaccine.",
            "mappings": {
                1: "Very Low",
                2: "Somewhat low",
                3: "Don't know",
                4: "Somewhat high",
                5: "Very high",
                999: "Missing"
            }
        },
        "opinion_seas_sick_from_vacc": {
            "description": "Respondent's worry of getting sick from taking seasonal flu vaccine.",
            "mappings": {
                1: "Not at all worried",
                2: "Not very worried",
                3: "Don't know",
                4: "Somewhat worried",
                5: "Very worried",
                999: "Missing"
            }
        },
        "age_group": {
            "description": "Age group of respondent.",
            "mappings": {
                None
            }
        },
        "education": {
            "description": "Self-reported education level.",
            "mappings": {
                None
            }
        },
        "ethnicity": {
            "description": "Ethnicity of respondent.",
            "mappings": {
                None
            }
        },
        "sex": {
            "description": "Sex of respondent.",
            "mappings": {
                None
            }
        },
        "income_poverty": {
            "description": "Household annual income of respondent with respect to 2008 Census poverty thresholds.",
            "mappings": {
                None
            }
        },
        "marital_status": {
            "description": "Marital status of respondent.",
            "mappings": {
                None
            }
        },
        "rent_or_own": {
            "description": "Housing situation of respondent.",
            "mappings": {
                None
            }
        },
        "employment_status": {
            "description": "Employment status of respondent.",
            "mappings": {
                None
            }
        },
        "hhs_geo_region": {
            "description": "Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services.",
            "mappings": {
                None
            }
        },
        "census_msa": {
            "description": "Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.",
            "mappings": {
                None
            }
        },
        "household_adults": {
            "description": "Number of other adults in household, top-coded to 3.",
            "mappings": {
                None
            }
        },
        "household_children": {
            "description": "Number of children in household, top-coded to 3.",
            "mappings": {
                None
            }
        },
        "employment_industry": {
            "description": "Type of industry respondent is employed in.",
            "mappings": {
                None
            }
        },
        "employment_occupation": {
            "description": "Type of occupation of respondent.",
            "mappings": {
                None
            }
        }
    }

    def get_info(var):
        string = "Definition: " + data_dictionary[var]["description"] + (" | Mappings: " + str(data_dictionary[var]["mappings"]) if data_dictionary[var]["mappings"] != {None} else "")
        return string

    # Sidebar with options
    st.sidebar.header("Filter Options")

    # Target Feature
    st.sidebar.write("Target Features")
    h1n1_vaccine        = st.sidebar.multiselect("H1N1 Vaccine",        np.unique(df["h1n1_vaccine"]),     np.unique(df["h1n1_vaccine"]),     help = get_info("h1n1_vaccine"),      format_func = lambda x: data_dictionary["h1n1_vaccine"]["mappings"][x])
    seasonal_vaccine    = st.sidebar.multiselect("Seasonal Vaccine",    np.unique(df["seasonal_vaccine"]), np.unique(df["seasonal_vaccine"]), help = get_info("seasonal_vaccine"),  format_func = lambda x: data_dictionary["seasonal_vaccine"]["mappings"][x])

    # Protected Characteristics
    protected_chars = st.sidebar.expander("Protected Characteristics")
    with protected_chars:
        age_group               = st.multiselect("Age Group",                   df["age_group"].unique().astype(str),               df["age_group"].unique().astype(str),               help = get_info("age_group"))
        education               = st.multiselect("Education",                   df["education"].unique().astype(str),               df["education"].unique().astype(str),               help = get_info("education"))
        ethnicity               = st.multiselect("Ethnicity",                   df["ethnicity"].unique().astype(str),               df["ethnicity"].unique().astype(str),               help = get_info("ethnicity"))
        sex                     = st.multiselect("Sex",                         df["sex"].unique().astype(str),                     df["sex"].unique().astype(str),                     help = get_info("sex"))
        chronic_med_condition   = st.multiselect("Chronic Medical Condition",   np.unique(df["chronic_med_condition"]),             np.unique(df["chronic_med_condition"]),             help = get_info("chronic_med_condition"), format_func = lambda x: data_dictionary["chronic_med_condition"]["mappings"][x])
        income_poverty          = st.multiselect("Income Poverty",              df["income_poverty"].unique().astype(str),          df["income_poverty"].unique().astype(str),          help = get_info("income_poverty"))

    opinion_chars = st.sidebar.expander("Opinion Characteristics")
    with opinion_chars:
        opinion_h1n1_vacc_effective = st.multiselect("H1N1 Effective",      np.unique(df["opinion_h1n1_vacc_effective"]),  np.unique(df["opinion_h1n1_vacc_effective"]),  help = get_info("opinion_h1n1_vacc_effective"),   format_func = lambda x: data_dictionary["opinion_h1n1_vacc_effective"]["mappings"][x])
        opinion_h1n1_risk           = st.multiselect("H1N1 Risk",           np.unique(df["opinion_h1n1_risk"]),            np.unique(df["opinion_h1n1_risk"]),            help = get_info("opinion_h1n1_risk"),             format_func = lambda x: data_dictionary["opinion_h1n1_risk"]["mappings"][x])
        opinion_h1n1_sick_from_vacc = st.multiselect("H1N1 Sick",           np.unique(df["opinion_h1n1_sick_from_vacc"]),  np.unique(df["opinion_h1n1_sick_from_vacc"]),  help = get_info("opinion_h1n1_sick_from_vacc"),   format_func = lambda x: data_dictionary["opinion_h1n1_sick_from_vacc"]["mappings"][x])
        opinion_seas_vacc_effective = st.multiselect("Seasonal Effective",  np.unique(df["opinion_seas_vacc_effective"]),  np.unique(df["opinion_seas_vacc_effective"]),  help = get_info("opinion_seas_vacc_effective"),   format_func = lambda x: data_dictionary["opinion_seas_vacc_effective"]["mappings"][x])
        opinion_seas_risk           = st.multiselect("Seasonal Risk",       np.unique(df["opinion_seas_risk"]),            np.unique(df["opinion_seas_risk"]),            help = get_info("opinion_seas_risk"),             format_func = lambda x: data_dictionary["opinion_seas_risk"]["mappings"][x])
        opinion_seas_sick_from_vacc = st.multiselect("Seasonal Sick",       np.unique(df["opinion_seas_sick_from_vacc"]),  np.unique(df["opinion_seas_sick_from_vacc"]),  help = get_info("opinion_seas_sick_from_vacc"),   format_func = lambda x: data_dictionary["opinion_seas_sick_from_vacc"]["mappings"][x])

    behavioural_chars = st.sidebar.expander("Behavioural Characteristics")
    with behavioural_chars:
        behavioral_antiviral_meds   = st.multiselect("Antiviral Medication",    np.unique(df["behavioral_antiviral_meds"]),    np.unique(df["behavioral_antiviral_meds"]),    help = get_info("behavioral_antiviral_meds"),     format_func = lambda x: data_dictionary["behavioral_antiviral_meds"]["mappings"][x])
        behavioral_avoidance        = st.multiselect("Avoidance",               np.unique(df["behavioral_avoidance"]),         np.unique(df["behavioral_avoidance"]),         help = get_info("behavioral_avoidance"),          format_func = lambda x: data_dictionary["behavioral_avoidance"]["mappings"][x])
        behavioral_face_mask        = st.multiselect("Face Mask",               np.unique(df["behavioral_face_mask"]),         np.unique(df["behavioral_face_mask"]),         help = get_info("behavioral_face_mask"),          format_func = lambda x: data_dictionary["behavioral_face_mask"]["mappings"][x])
        behavioral_wash_hands       = st.multiselect("Wash Hande",              np.unique(df["behavioral_wash_hands"]),        np.unique(df["behavioral_wash_hands"]),        help = get_info("behavioral_wash_hands"),         format_func = lambda x: data_dictionary["behavioral_wash_hands"]["mappings"][x])
        behavioral_large_gatherings = st.multiselect("Large Gatherings",        np.unique(df["behavioral_large_gatherings"]),  np.unique(df["behavioral_large_gatherings"]),  help = get_info("behavioral_large_gatherings"),   format_func = lambda x: data_dictionary["behavioral_large_gatherings"]["mappings"][x])
        behavioral_outside_home     = st.multiselect("Outside Home",            np.unique(df["behavioral_outside_home"]),      np.unique(df["behavioral_outside_home"]),      help = get_info("behavioral_outside_home"),       format_func = lambda x: data_dictionary["behavioral_outside_home"]["mappings"][x])
        behavioral_touch_face       = st.multiselect("Touch Face",              np.unique(df["behavioral_touch_face"]),        np.unique(df["behavioral_touch_face"]),        help = get_info("behavioral_touch_face"),         format_func = lambda x: data_dictionary["behavioral_touch_face"]["mappings"][x])
        
    health_chars = st.sidebar.expander("Health Characteristics")
    with health_chars:
        h1n1_concern            = st.multiselect("H1N1 Concern",        np.unique(df["h1n1_concern"]),         np.unique(df["h1n1_concern"]),         help = get_info("h1n1_concern"),          format_func = lambda x: data_dictionary["h1n1_concern"]["mappings"][x])
        h1n1_knowledge          = st.multiselect("H1N1 Knowledge",      np.unique(df["h1n1_knowledge"]),       np.unique(df["h1n1_knowledge"]),       help = get_info("h1n1_knowledge"),        format_func = lambda x: data_dictionary["h1n1_knowledge"]["mappings"][x])
        doctor_recc_h1n1        = st.multiselect("Recommend H1N1",      np.unique(df["doctor_recc_h1n1"]),     np.unique(df["doctor_recc_h1n1"]),     help = get_info("doctor_recc_h1n1"),      format_func = lambda x: data_dictionary["doctor_recc_h1n1"]["mappings"][x])
        doctor_recc_seasonal    = st.multiselect("Recommend Seasonal",  np.unique(df["doctor_recc_seasonal"]), np.unique(df["doctor_recc_seasonal"]), help = get_info("doctor_recc_seasonal"),  format_func = lambda x: data_dictionary["doctor_recc_seasonal"]["mappings"][x])
        health_worker           = st.multiselect("Health Worker",       np.unique(df["health_worker"]),        np.unique(df["health_worker"]),        help = get_info("health_worker"),         format_func = lambda x: data_dictionary["health_worker"]["mappings"][x])
        health_insurance        = st.multiselect("Health Insurance",    np.unique(df["health_insurance"]),     np.unique(df["health_insurance"]),     help = get_info("health_insurance"),      format_func = lambda x: data_dictionary["health_insurance"]["mappings"][x])

    # Other Characteristics
    other_chars = st.sidebar.expander("Other Characteristics")
    with other_chars:
        child_under_6_months    = st.multiselect("Young Child",             np.unique(df["child_under_6_months"]),              np.unique(df["child_under_6_months"]),                                                                              help = get_info("child_under_6_months"), format_func = lambda x: data_dictionary["child_under_6_months"]["mappings"][x])
        marital_status          = st.multiselect("Marital Status",          df["marital_status"].unique().astype(str),          df["marital_status"].unique().astype(str),                                                                          help = get_info("marital_status"))
        rent_or_own             = st.multiselect("Rent or Own",             df["rent_or_own"].unique().astype(str),             df["rent_or_own"].unique().astype(str),                                                                             help = get_info("rent_or_own"))
        employment_status       = st.multiselect("Employment Status",       df["employment_status"].unique().astype(str),       df["employment_status"].unique().astype(str),                                                                       help = get_info("employment_status"))
        # hhs_geo_region          = st.multiselect("HHS Geo Region",          df["hhs_geo_region"].unique().astype(str),          df["hhs_geo_region"].unique().astype(str),                                                                          help = get_info("hhs_geo_region"))
        census_msa              = st.multiselect("Census MSA",              df["census_msa"].unique().astype(str),              df["census_msa"].unique().astype(str),                                                                              help = get_info("census_msa"))
        household_adults        = st.slider("Household Adults",             int(df["household_adults"].min()),                  int(df["household_adults"].max()), (int(df["household_adults"].min()), int(df["household_adults"].max())),          help = get_info("household_adults"))
        household_children      = st.slider("Household Children",           int(df["household_children"].min()),                int(df["household_children"].max()), (int(df["household_children"].min()), int(df["household_children"].max())),    help = get_info("household_children"))
        # employment_industry     = st.multiselect("Employment Industry",     df["employment_industry"].unique().astype(str),     df["employment_industry"].unique().astype(str),                                                                     help = get_info("employment_industry"))
        # employment_occupation   = st.multiselect("Employment Occupation",   df["employment_occupation"].unique().astype(str),   df["employment_occupation"].unique().astype(str),                                                                   help = get_info("employment_occupation"))

        
    filtered_df = df[
        # Target
        (df["h1n1_vaccine"].isin(h1n1_vaccine)) &
        (df["seasonal_vaccine"].isin(seasonal_vaccine)) &
        # Protected Characteristics
        (df["age_group"].isin(age_group)) &
        (df["education"].isin(education)) &
        (df["ethnicity"].isin(ethnicity)) &
        (df["sex"].isin(sex)) &
        (df["chronic_med_condition"].isin(chronic_med_condition)) &
        (df["income_poverty"].isin(income_poverty)) &
        # Opinion Characteristics
        (df["opinion_h1n1_vacc_effective"].isin(opinion_h1n1_vacc_effective)) &
        (df["opinion_h1n1_risk"].isin(opinion_h1n1_risk)) &
        (df["opinion_h1n1_sick_from_vacc"].isin(opinion_h1n1_sick_from_vacc)) &
        (df["opinion_seas_vacc_effective"].isin(opinion_seas_vacc_effective)) &
        (df["opinion_seas_risk"].isin(opinion_seas_risk)) &
        (df["opinion_seas_sick_from_vacc"].isin(opinion_seas_sick_from_vacc)) &
        # Behavioural Characteristics
        (df["behavioral_antiviral_meds"].isin(behavioral_antiviral_meds)) &
        (df["behavioral_avoidance"].isin(behavioral_avoidance)) &
        (df["behavioral_face_mask"].isin(behavioral_face_mask)) &
        (df["behavioral_wash_hands"].isin(behavioral_wash_hands)) &
        (df["behavioral_large_gatherings"].isin(behavioral_large_gatherings)) &
        (df["behavioral_outside_home"].isin(behavioral_outside_home)) &
        (df["behavioral_touch_face"].isin(behavioral_touch_face)) &
        # Health Characteristics
        (df["h1n1_concern"].isin(h1n1_concern)) &
        (df["h1n1_knowledge"].isin(h1n1_knowledge)) &
        (df["doctor_recc_h1n1"].isin(doctor_recc_h1n1)) &
        (df["doctor_recc_seasonal"].isin(doctor_recc_seasonal)) &
        (df["health_worker"].isin(health_worker)) &
        (df["health_insurance"].isin(health_insurance)) &
        # Other Characteristics
        (df["child_under_6_months"].isin(child_under_6_months)) &
        (df["marital_status"].isin(marital_status)) &
        (df["rent_or_own"].isin(rent_or_own)) &
        (df["employment_status"].isin(employment_status)) &
        # (df["hhs_geo_region"].isin(hhs_geo_region)) &
        (df["census_msa"].isin(census_msa)) &
        (df["household_adults"] >= household_adults[0]) & (df["household_adults"] <= household_adults[1]) &
        (df["household_children"] >= household_children[0]) & (df["household_children"] <= household_children[1]) # &
        # (df["employment_industry"].isin(employment_industry)) &
        #(df["employment_industry"].isin(employment_industry))
    ]

    data_dict = st.expander(":books: Data Dictionary")
    with data_dict:
        st.markdown('''
                    ##### **Data Labels**

                    Each row in the dataset represents one person who responded to the National 2009 H1N1 Flu Survey.

                    There are two target variables:

                    * **:red-background[h1n1_vaccine]** - Whether respondent received H1N1 flu vaccine.

                    * **:red-background[seasonal_vaccine]** - Whether respondent received seasonal flu vaccine.

                    Both are binary variables: 0 = No; 1 = Yes. Some respondents didn't get either vaccine, others got only one, and some got both. This is formulated as a multilabel (and not multiclass) problem.

                    ##### **Features**

                    Provided is a dataset with 36 columns. The first column respondent_id is a unique and random identifier. The remaining 35 features are described below.

                    For all binary variables: 0 = No; 1 = Yes, 999 = Missing.

                    * **:red-background[h1n1_concern]** - Level of concern about the H1N1 flu.
                    0 = Not at all concerned; 1 = Not very concerned; 2 = Somewhat concerned; 3 = Very concerned.

                    * **:red-background[h1n1_knowledge]** - Level of knowledge about H1N1 flu.
                    0 = No knowledge; 1 = A little knowledge; 2 = A lot of knowledge.

                    * behavioral_antiviral_meds]** - Has taken antiviral medications. (binary)

                    * **:red-background[behavioral_avoidance]** - Has avoided close contact with others with flu-like symptoms. (binary)

                    * **:red-background[behavioral_face_mask]** - Has bought a face mask. (binary)

                    * **:red-background[behavioral_wash_hands]** - Has frequently washed hands or used hand sanitizer. (binary)

                    * **:red-background[behavioral_large_gatherings]** - Has reduced time at large gatherings. (binary)

                    * **:red-background[behavioral_outside_home]** - Has reduced contact with people outside of own household. (binary)

                    * **:red-background[behavioral_touch_face]** - Has avoided touching eyes, nose, or mouth. (binary)

                    * **:red-background[doctor_recc_h1n1]** - H1N1 flu vaccine was recommended by doctor. (binary)

                    * **:red-background[doctor_recc_seasonal]** - Seasonal flu vaccine was recommended by doctor. (binary)

                    * **:red-background[chronic_med_condition]** - Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or other anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronic illness or by medicines taken for a chronic illness. (binary)

                    * **:red-background[child_under_6_months]** - Has regular close contact with a child under the age of six months. (binary)

                    * **:red-background[health_worker]** - Is a healthcare worker. (binary)

                    * **:red-background[health_insurance]** - Has health insurance. (binary)

                    * **:red-background[opinion_h1n1_vacc_effective]** - Respondent's opinion about H1N1 vaccine effectiveness.
                    1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.

                    * **:red-background[opinion_h1n1_risk]** - Respondent's opinion about risk of getting sick with H1N1 flu without vaccine.
                    1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.

                    * **:red-background[opinion_h1n1_sick_from_vacc]** - Respondent's worry of getting sick from taking H1N1 vaccine.
                    1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.

                    * **:red-background[opinion_seas_vacc_effective]** - Respondent's opinion about seasonal flu vaccine effectiveness.
                    1 = Not at all effective; 2 = Not very effective; 3 = Don't know; 4 = Somewhat effective; 5 = Very effective.

                    * **:red-background[opinion_seas_risk]** - Respondent's opinion about risk of getting sick with seasonal flu without vaccine.
                    1 = Very Low; 2 = Somewhat low; 3 = Don't know; 4 = Somewhat high; 5 = Very high.

                    * **:red-background[opinion_seas_sick_from_vacc]** - Respondent's worry of getting sick from taking seasonal flu vaccine.
                    1 = Not at all worried; 2 = Not very worried; 3 = Don't know; 4 = Somewhat worried; 5 = Very worried.

                    * **:red-background[age_group]** - Age group of respondent.

                    * **:red-background[education]** - Self-reported education level.

                    * **:red-background[ethnicity]** - Ethnicity of respondent.

                    * **:red-background[sex]** - Sex of respondent.

                    * **:red-background[income_poverty]** - Household annual income of respondent with respect to 2008 Census poverty thresholds.

                    * **:red-background[marital_status]** - Marital status of respondent.

                    * **:red-background[rent_or_own]** - Housing situation of respondent.

                    * **:red-background[employment_status]** - Employment status of respondent.

                    * **:red-background[hhs_geo_region]** - Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.

                    * **:red-background[census_msa]** - Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census.

                    * **:red-background[household_adults]** - Number of other adults in household, top-coded to 3.

                    * **:red-background[household_children]** - Number of children in household, top-coded to 3.

                    * **:red-background[employment_industry]** - Type of industry respondent is employed in. Values are represented as short random character strings.

                    * **:red-background[employment_occupation]** - Type of occupation of respondent. Values are represented as short random character strings.
                    ''')

    # Display the dataset


    st.write("### Inspecting the Data")
    st.dataframe(filtered_df)

    st.write("### Data Summary")
    var_type = st.radio("Select Class Type", [":1234: Numeric", ":abc: Categoric"], horizontal = True)

    def cat_describe(data):
        cat_summary = {}
        for var in data.select_dtypes("category").columns:
            freq = pd.DataFrame(data[data[var] != "Missing"][var].value_counts().reset_index().sort_values(by = "count", ascending = False))
            freq["percent"] = 100 * freq["count"] / freq["count"].sum()

            unique = len(data[var].unique()) - (1 if any(df[var] == "Missing") else 0) # exclude missing
            missing = sum(data[var] == "Missing")
            missing_p = 100 * missing / len(data)
            cat_summary.update({var: {
                "unique": unique,
                "most_freq": f"{freq.iloc[0, 0]} | {freq.iloc[0, 1]:,} ({freq.iloc[0, 2]:.2f}%)",
                "least_freq": f"{freq.iloc[-2, 0]} | {freq.iloc[-2, 1]} ({freq.iloc[-2, 2]:.2f}%)",
                "missing": missing,
                "missing%":  round(missing_p, 2)
            }})
        return pd.DataFrame(cat_summary).T

    def prettyDescribe(data, round_n = 2):
        if var_type == ":1234: Numeric":
            describe_data = data.mask(df.isin([999, "Missing"])).describe().drop(["count", "25%", "50%", "75%"], axis = 0).T
            describe_data["missing"] = data.mask(data.isin([999, "Missing"])).isna().agg(sum)
            describe_data["missing%"] = 100 * describe_data["missing"] / len(df)
            return round(describe_data, round_n)
        elif var_type == ":abc: Categoric":
            return cat_describe(data)
        
    st.dataframe(prettyDescribe(filtered_df))
    st.write(f"{len(filtered_df):,} observations | {len(df) - len(filtered_df):,} ({100 * (len(df) - len(filtered_df))/len(df):.1f}%) of observations filtered out")

    st.write("### Distributions of Vaccination Indicators")
    
    # spinner = st.spinner(text = "Generating plots...")
    # with spinner:
    responses = ["h1n1_vaccine", "seasonal_vaccine"]
    features = [x for x in df.columns if x not in ["respondent_id", "h1n1_vaccine", "seasonal_vaccine"]]

    percent_complete = 0
    progress_text = "Generating plots..."
    increment = 1/len(features)
    my_bar = st.progress(percent_complete, text = progress_text)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21, tab22, tab23, tab24, tab25, tab26, tab27, tab28, tab29, tab30, tab31, tab32, tab33, tab34, tab35 = st.tabs(features)
    tabs = [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21, tab22, tab23, tab24, tab25, tab26, tab27, tab28, tab29, tab30, tab31, tab32, tab33, tab34, tab35]

    def generate_plot_data(feature, target, filtered = True):
        if filtered:
            to_plot = filtered_df[[target, feature]].sort_values(by = [target, feature], ascending = True)
        else:
            to_plot = df[[target, feature]].sort_values(by = [target, feature], ascending = True)
        to_plot = to_plot[~to_plot[feature].isin([999, "Missing"])]
        to_plot = to_plot.value_counts(subset = [target, feature]).to_frame().reset_index()
        to_plot["percent"] = to_plot["count"] / to_plot.groupby(feature)["count"].transform("sum")
        return to_plot.sort_values(by = [target, feature], ascending = True).reset_index(drop = True)

    update_legend = {"0": "No", "1": "Yes"}
    data_distributions = {}
    
    for feature in features:
        fig = make_subplots(rows = 2, cols = 2, shared_xaxes = True,
                            subplot_titles = ["H1N1 Vaccine","H1N1 Vaccine",
                                            "Seasonal Vaccine", "Seasonal Vaccine"])

        h1n1 = generate_plot_data(feature, "h1n1_vaccine")
        seasonal = generate_plot_data(feature, "seasonal_vaccine")

        fig1 = px.bar(h1n1, x = feature, y = "count", text_auto = ",",
                    color = h1n1["h1n1_vaccine"].astype(str), 
                    color_discrete_map = {"0": "#EF553B", "1": "#636EFA"})
        
        fig2 = px.bar(h1n1, x = feature, y = "percent", text_auto = ".1f",
                    color = h1n1["h1n1_vaccine"].astype(str),
                    color_discrete_map = {"0": "#EF553B", "1": "#636EFA"}).update_traces(showlegend = False)
        
        fig3 = px.bar(seasonal, x = feature, y = "count", text_auto = ",",
                    color = seasonal["seasonal_vaccine"].astype(str), 
                    color_discrete_map = {"0": "#EF553B", "1": "#636EFA"}).update_traces(showlegend = False)
        
        fig4 = px.bar(seasonal, x = feature, y = "percent", text_auto = ".1f",
                    color = seasonal["seasonal_vaccine"].astype(str), 
                    color_discrete_map = {"0": "#EF553B", "1": "#636EFA"}).update_traces(showlegend = False)

        for trace in fig1.data:
            fig.add_trace(trace, 1, 1)
        for trace in fig2.data:
            fig.add_trace(trace, 1, 2)
        for trace in fig3.data:
            fig.add_trace(trace, 2, 1)
        for trace in fig4.data:
            fig.add_trace(trace, 2, 2)
        
        fig.update_layout(barmode = "stack", title = {"text": feature.title() + "<br><sup>Histogram & 100% Stacked Bar Chart</sup>"},
                        legend_title_text = "Vaccinated")
        fig.for_each_trace(lambda t: t.update(name = update_legend[t.name],
                                        legendgroup = update_legend[t.name],
                                        hovertemplate = t.hovertemplate.replace(t.name, update_legend[t.name])))
        if data_dictionary[feature]["mappings"] != {None}:
            update_dict = dict(tickvals = list(data_dictionary[feature]["mappings"].keys()), ticktext = list(data_dictionary[feature]["mappings"].values()))
            fig.update_layout(xaxis3 = update_dict, xaxis4 = update_dict)
        data_distributions.update({feature: fig})
        percent_complete += increment
        my_bar.progress(percent_complete, text = progress_text)

    # Plotly
    # Plot distributions
    my_bar.progress(percent_complete, text = "Plotting...")
    for tab, var in zip(tabs, features):
        with tab:
            st.write(f"### {var}")
            st.write(f"{data_dictionary[var]['description']}")
            st.plotly_chart(data_distributions[var])
    my_bar.empty()

with data_documentation:
    st.markdown('''
                ##### **Data Source**
                
                The data comes from the National 2009 H1N1 Flu Survey (NHFS) and is provided courtesy of the United States [National Center for Health Statistics](https://www.cdc.gov/nchs/index.htm) :link:.

                In their own words:

                The National 2009 H1N1 Flu Survey (NHFS) was sponsored by the National Center for Immunization and Respiratory Diseases (NCIRD) and conducted jointly by NCIRD and the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC). The NHFS was a list-assisted random-digit-dialing telephone survey of households, designed to monitor influenza immunization coverage in the 2009-10 season.

                The target population for the NHFS was all persons 6 months or older living in the United States at the time of the interview. Data from the NHFS were used to produce timely estimates of vaccination coverage rates for both the monovalent pH1N1 and trivalent seasonal influenza vaccines.

                The NHFS was conducted between October 2009 and June 2010. It was one-time survey designed specifically to monitor vaccination during the 2009-2010 flu season in response to the 2009 H1N1 pandemic. The CDC has other ongoing programs for annual phone surveys that continue to monitor seasonal flu vaccination.
                
                ---

                ##### **Data Restrictions**

                The source dataset comes with the following data use restrictions:

                * The Public Health Service Act (Section 308(d)) provides that the data collected by the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC), may be used only for the purpose of health statistical reporting and analysis.

                * Any effort to determine the identity of any reported case is prohibited by this law.

                * NCHS does all it can to ensure that the identity of data subjects cannot be disclosed. All direct identifiers, as well as any characteristics that might lead to identification, are omitted from the data files. Any intentional identification or disclosure of a person or establishment violates the assurances of confidentiality given to the providers of the information.

                Therefore, users will:

                * Use the data in these data files for statistical reporting and analysis only.

                * Make no use of the identity of any person or establishment discovered inadvertently and advise the Director, NCHS, of any such discovery (1 (800) 232-4636).

                * Not link these data files with individually identifiable data from other NCHS or non-NCHS data files.

                * By using these data, you signify your agreement to comply with the above requirements.

                ''')