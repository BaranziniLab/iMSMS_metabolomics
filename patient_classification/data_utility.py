import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from paths import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder




def load_global_data(sample):
    if sample == "serum":
        filename = GLOBAL_SERUM_DATA_FILENAME
    else:
        filename = GLOBAL_STOOL_DATA_FILENAME    
    file_path = os.path.join(DATA_ROOT_PATH, filename)
    sheet_name = ["Chemical Annotation", "Sample Meta Data", "Log Transformed Data"]
    analyte_metadata = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[0])
    patient_metadata = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[1])
    data = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[2])
    global_metabolomics_compound_spoke_map = pd.read_csv(os.path.join(OUTPUT_PATH, "global_metabolomics_compound_spoke_map.csv"))
    analyte_columns = list(data.columns)
    analyte_columns.remove("PARENT_SAMPLE_NAME")
    analyte_columns_selected = global_metabolomics_compound_spoke_map[global_metabolomics_compound_spoke_map.CHEM_ID.isin(analyte_columns)]["CHEM_ID"].unique()
    data_with_analyte_columns_selected = data[["PARENT_SAMPLE_NAME"]+list(analyte_columns_selected)]
    selected_metadata_columns = ["PARENT_SAMPLE_NAME", "CLIENT_IDENTIFIER", "GROUP_NAME", "AGE", "BMI", "GENDER", "CLIENT_SAMPLE_ID", "CLIENT_MATRIX", "TREATMENT"]
    patient_metadata_selected_columns = patient_metadata[selected_metadata_columns]
    patient_metadata_selected_columns.loc[:, 'house'] = (patient_metadata_selected_columns['CLIENT_SAMPLE_ID'].str[:3] + patient_metadata_selected_columns['CLIENT_SAMPLE_ID'].str[-4:])
    patient_metadata_selected_columns.loc[:, 'site'] = patient_metadata_selected_columns.loc[:, 'CLIENT_SAMPLE_ID'].str[:3]
    selected_metadata_columns = selected_metadata_columns + ["house", "site"]
    data_with_patient_metadata = pd.merge(data_with_analyte_columns_selected, patient_metadata_selected_columns, on="PARENT_SAMPLE_NAME")
    data_with_patient_metadata_reverse_log_transformed = data_with_patient_metadata.copy()
    for analyte in analyte_columns_selected:
        analyte_concentration = data_with_patient_metadata_reverse_log_transformed[analyte].values
        data_with_patient_metadata_reverse_log_transformed.loc[:, analyte] = np.exp(analyte_concentration)    
    house_to_exclude = data_with_patient_metadata_reverse_log_transformed[data_with_patient_metadata_reverse_log_transformed.isna().any(axis=1)].house.values
    data_with_patient_metadata_reverse_log_transformed_nan_removed = data_with_patient_metadata_reverse_log_transformed[~data_with_patient_metadata_reverse_log_transformed["house"].isin(house_to_exclude)]
#     metadata_df = data_with_patient_metadata_reverse_log_transformed_nan_removed[selected_metadata_columns]
#     feature_vectors = data_with_patient_metadata_reverse_log_transformed_nan_removed[analyte_columns_selected].values
    return data_with_patient_metadata_reverse_log_transformed_nan_removed, analyte_columns_selected


def get_processed_global_data(sample):
    data_with_patient_metadata_reverse_log_transformed_nan_removed, analyte_columns_selected = load_global_data(sample)
    data_to_go = data_with_patient_metadata_reverse_log_transformed_nan_removed.copy()
    le = LabelEncoder()
    data_to_go.loc[:, 'Disease_label'] = le.fit_transform(data_to_go['GROUP_NAME'])
    return data_to_go, analyte_columns_selected


def get_processed_targeted_data(sample):
    filename = SHORT_CHAIN_FATTY_ACID_DATA_FILENAME
    mapping_filename = "short_chain_fatty_acid_spoke_map.csv"
    file_path = os.path.join(DATA_ROOT_PATH, filename)
    mapping_filepath = os.path.join(OUTPUT_PATH, mapping_filename)
    data = pd.read_excel(file_path, engine='openpyxl')
    data = data[data["Client Matrix"]==sample]
    mapping_data = pd.read_csv(mapping_filepath)
    analytes = mapping_data["name"].unique()
    data.loc[:, 'house'] = (data['Client Sample ID'].str[:3] + data['Client Sample ID'].str[-4:])
    data_exclude_outlier_threshold_column = data.drop("Analysis Comment", axis=1)
    house_to_exclude = data_exclude_outlier_threshold_column[data_exclude_outlier_threshold_column.isna().any(axis=1)].house.values
    data_nan_removed = data[~data["house"].isin(house_to_exclude)]
    le = LabelEncoder()
    data_nan_removed.loc[:, 'Disease_label'] = le.fit_transform(data_nan_removed['Group Name'])
    analyte_features = pd.DataFrame(columns=data_nan_removed.Analyte.unique())
    for item in data_nan_removed.Analyte.unique():
        analyte_features.loc[:, item] = data_nan_removed[data_nan_removed.Analyte == item].Result.values
    analyte_features.loc[:, "Disease_label"] = data_nan_removed[data_nan_removed.Analyte == item].Disease_label.values
    analyte_features.loc[:, "CLIENT_SAMPLE_ID"] = data_nan_removed[data_nan_removed.Analyte == item]["Client Sample ID"].values
    return analyte_features, data_nan_removed.Analyte.unique()
    
    
def get_processed_global_and_targeted_data(sample):
    data_global, global_analyte_columns_selected = get_processed_global_data(sample)
    data_targeted, targeted_analyte_columns_selected  = get_processed_targeted_data(sample)
    data_global_targeted_merge = pd.merge(data_global, data_targeted, on="CLIENT_SAMPLE_ID").drop("Disease_label_y", axis=1).rename(columns={"Disease_label_x":"Disease_label"})
    analyte_columns_selected = np.concatenate([global_analyte_columns_selected, targeted_analyte_columns_selected], axis=0)
    return data_global_targeted_merge, analyte_columns_selected 
    