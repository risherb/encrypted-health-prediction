"""
In this file, we roughly split up a list of symptoms, taken from "./training.csv" file, avalaible
through: "https://github.com/anujdutt9/Disease-Prediction-from-Symptoms/tree/master/dataset"
into medical categories, in order to make the UI more plesant for the users.

Each variable contains a list of symptoms sthat can be pecific to a part of the body or to a list
of similar symptoms.
"""


DIGESTIVE_SYSTEM_SYMPTOMS = {
    "DIGESTIVE_SYSTEM_CONCERNS": [
        "stomach_pain",
        "acidity",
        "vomiting",
        "indigestion",
        "constipation",
        "abdominal_pain",
        "diarrhea",
        "nausea",
        "distention_of_abdomen",
        "stomach_bleeding",
        "pain_during_bowel_movements",
        "passage_of_gases",
        "red_spots_over_body",
        "swelling_of_stomach",
        "bloody_stool",
        "irritation_in_anus",
        "pain_in_anal_region",
        "abnormal_menstruation",
    ]
}

DERMATOLOGICAL_SYMPTOMS = {
    "DERMATOLOGICAL_CONCERNS": [
        "itching",
        "skin_rash",
        "pus_filled_pimples",
        "blackheads",
        "scurving",
        "skin_peeling",
        "silver_like_dusting",
        "small_dents_in_nails",
        "inflammatory_nails",
        "blister",
        "red_sore_around_nose",
        "bruising",
        "yellow_crust_ooze",
        "dischromic_patches",
        "nodal_skin_eruptions",
        "toxic_look_(typhus)",
        "brittle_nails",
        "yellowish_skin",
    ]
}

ORL_SYMPTOMS = {
    "ORL_CONCERNS": [
        "loss_of_smell",
        "continuous_sneezing",
        "runny_nose",
        "patches_in_throat",
        "throat_irritation",
        "sinus_pressure",
        "enlarged_thyroid",
        "loss_of_balance",
        "unsteadiness",
        "dizziness",
        "spinning_movements",
    ]
}

THORAX_SYMPTOMS = {
    "THORAX_CONCERNS": [
        "breathlessness",
        "chest_pain",
        "cough",
        "rusty_sputum",
        "phlegm",
        "mucoid_sputum",
        "congestion",
        "blood_in_sputum",
        "fast_heart_rate",
    ]
}

OPHTHALMOLOGICAL_SYMPTOMS = {
    "OPHTHALMOLOGICAL_CONCERNS": [
        "sunken_eyes",
        "redness_of_eyes",
        "watering_from_eyes",
        "blurred_and_distorted_vision",
        "pain_behind_the_eyes",
        "visual_disturbances",
    ]
}

VASCULAR_LYMPHATIC_SYMPTOMS = {
    "VASCULAR_AND_LYMPHATIC_CONCERNS": [
        "cold_hands_and_feets",
        "swollen_blood_vessels",
        "swollen_legs",
        "swelled_lymph_nodes",
        "palpitations",
        "prominent_veins_on_calf",
        "yellowing_of_eyes",
        "puffy_face_and_eyes",
        "severe_fluid_overload",
        "swollen_extremeties",
    ]
}

UROLOGICAL_SYMPTOMS = {
    "UROLOGICAL_CONCERNS": [
        "burning_micturition",
        "spotting_urination",
        "yellow_urine",
        "bladder_discomfort",
        "foul_smell_of_urine",
        "continuous_feel_of_urine",
        "polyuria",
        "dark_urine",
    ]
}

MUSCULOSKELETAL_SYMPTOMS = {
    "MUSCULOSKELETAL_CONCERNS": [
        "joint_pain",
        "muscle_wasting",
        "muscle_pain",
        "muscle_weakness",
        "knee_pain",
        "stiff_neck",
        "swelling_joints",
        "movement_stiffness",
        "hip_joint_pain",
        "painful_walking",
        "weakness_of_one_body_side",
        "neck_pain",
        "back_pain",
        "weakness_in_limbs",
        "cramps",
    ]
}

GENERAL_SYMPTOMS = {
    "GENERAL_CONCERNS": [
        "acute_liver_failure",
        "anxiety",
        "restlessness",
        "lethargy",
        "mood_swings",
        "irritability",
        "lack_of_concentration",
        "fatigue",
        "malaise",
        "weight_gain",
        "increased_appetite",
        "weight_loss",
        "loss_of_appetite",
        "excess_body_fat",
        "excessive_hunger",
        "ulcers_on_tongue",
        "shivering",
        "chills",
        "irregular_sugar_level",
        "high_fever",
        "slurred_speech",
        "sweating",
        "internal_itching",
        "mild_fever",
        "dehydration",
        "headache",
        "frequent_unprotected_sexual_intercourse_with_multiple_partners",
        "drying_and_tingling_lips",
        "altered_sensorium",
        "family_history",
        "receiving_blood_transfusion",
        "receiving_unsterile_injections",
        "chronic_alcohol_abuse",
    ]
}

SYMPTOMS_LIST = [
    # Column 1
    DIGESTIVE_SYSTEM_SYMPTOMS,
    UROLOGICAL_SYMPTOMS,
    VASCULAR_LYMPHATIC_SYMPTOMS,
    # Column 2
    ORL_SYMPTOMS,
    DERMATOLOGICAL_SYMPTOMS,
    MUSCULOSKELETAL_SYMPTOMS,
    # Column 3
    OPHTHALMOLOGICAL_SYMPTOMS,
    THORAX_SYMPTOMS,
    GENERAL_SYMPTOMS,
]
