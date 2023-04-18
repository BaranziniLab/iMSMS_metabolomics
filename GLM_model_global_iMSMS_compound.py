from paths import *
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import multiprocessing as mp
import sys
import time
import joblib

sample = sys.argv[1]
data_type = sys.argv[2]
NCORES = int(sys.argv[3])


def glm_model(data_, analyte_, sample_, remove_outlier):
    data_analyte = data_[selected_metadata_columns+[analyte_, "site", "house"]]
    if remove_outlier:
        data_analyte = remove_outliers(data_analyte, analyte_)        
    le = LabelEncoder()
    data_analyte.loc[:, 'Disease_label'] = le.fit_transform(data_analyte['GROUP_NAME'])
    data_analyte.loc[:, 'Gender_label'] = le.fit_transform(data_analyte['GENDER'])
    data_analyte = data_analyte[[analyte_, "Disease_label", "AGE", "BMI", "Gender_label", "site", "house"]]
    X = sm.add_constant(data_analyte[['AGE', 'BMI', 'Gender_label', 'Disease_label']])
    site_dummies = pd.get_dummies(data_analyte['site'], prefix='site', drop_first=True)
    house_dummies = pd.get_dummies(data_analyte['house'], prefix='house', drop_first=True)
    X = pd.concat([X, site_dummies, house_dummies], axis=1)
    try:
        model = sm.GLM(data_analyte[analyte_], X, family=sm.families.Gamma(link=sm.families.links.log()))
        mdf = model.fit()
    except:
        mdf = None
    out_dict = {}
    out_dict["analyte"] = analyte_
    out_dict["sample"] = sample_
    out_dict["patient_sample_count"] = data_analyte.shape[0]
    out_dict["model"] = mdf
    return out_dict 


def get_model_parallel(data_, analytes_, sample_, remove_outlier_flag):    
    data_list = [data_]*len(analytes_)
    analytes_list = list(analytes_)
    sample_list = [sample_]*len(analytes_)
    remove_outlier_flag_list = [remove_outlier_flag]*len(analytes_)
    arg_list = list(zip(data_list, analytes_list, sample_list, remove_outlier_flag_list))
    p = mp.Pool(NCORES)
    model_list = p.starmap(glm_model, arg_list)
    p.close()
    p.join()
    return model_list

#Detecting outliers using Tukey Fences method
#Ref: https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_summarizingdata/bs704_summarizingdata7.html#headingtaglink_3
def remove_outliers(data_, analyte):
    analyte_concentration_values = data_[analyte]
    # Calculate quartiles and IQR
    q1 = analyte_concentration_values.quantile(0.25)
    q3 = analyte_concentration_values.quantile(0.75)
    iqr = q3 - q1
    # Define Tukey's fences
    k = 1.5
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    # Filter non-outliers
    non_outliers = data_[(data_[analyte] >= lower_fence) & (data_[analyte] <= upper_fence)]
    house_counts = non_outliers.groupby("house").size()
    # Filter to keep only the rows with paired house IDs
    paired_houses = house_counts[house_counts == 2].index
    non_outliers = non_outliers[non_outliers["house"].isin(paired_houses)]
    return non_outliers


def load_data(sample):
    if sample == "serum":
        filename = GLOBAL_SERUM_DATA_FILENAME
    else:
        filename = GLOBAL_STOOL_DATA_FILENAME
    file_path = os.path.join(DATA_ROOT_PATH, filename)
    sheet_name = ["Chemical Annotation", "Sample Meta Data", "Log Transformed Data"]
    analyte_metadata = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[0])
    patient_metadata = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[1])
    data = pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name[2])
    return analyte_metadata, patient_metadata, data


def reverse_log_transform(data_, analyte_columns_selected_):
    data_with_patient_metadata_reverse_log_transformed = data_.copy()
    for analyte in analyte_columns_selected_:
        analyte_concentration = data_with_patient_metadata_reverse_log_transformed[analyte].values
        data_with_patient_metadata_reverse_log_transformed.loc[:, analyte] = np.exp(analyte_concentration)
    return data_with_patient_metadata_reverse_log_transformed


def main():
    global selected_metadata_columns
    start_time = time.time()
    analyte_metadata, patient_metadata, data = load_data(sample)
    global_metabolomics_compound_spoke_map = pd.read_csv(os.path.join(OUTPUT_PATH, "global_metabolomics_compound_spoke_map.csv"))
    analyte_columns = list(data.columns)
    analyte_columns.remove("PARENT_SAMPLE_NAME")
    analyte_columns_selected = global_metabolomics_compound_spoke_map[global_metabolomics_compound_spoke_map.CHEM_ID.isin(analyte_columns)]["CHEM_ID"].unique()
    data_with_analyte_columns_selected = data[["PARENT_SAMPLE_NAME"]+list(analyte_columns_selected)]
    selected_metadata_columns = ["PARENT_SAMPLE_NAME", "CLIENT_IDENTIFIER", "GROUP_NAME", "AGE", "BMI", "GENDER", "CLIENT_SAMPLE_ID", "CLIENT_MATRIX", "TREATMENT"]
    patient_metadata_selected_columns = patient_metadata[selected_metadata_columns]
    patient_metadata_selected_columns.loc[:, 'house'] = (patient_metadata_selected_columns['CLIENT_SAMPLE_ID'].str[:3] + patient_metadata_selected_columns['CLIENT_SAMPLE_ID'].str[-4:])
    patient_metadata_selected_columns.loc[:, 'site'] = patient_metadata_selected_columns.loc[:, 'CLIENT_SAMPLE_ID'].str[:3]
    le = LabelEncoder()
    patient_metadata_selected_columns.loc[:, 'Disease_label'] = le.fit_transform(patient_metadata_selected_columns['GROUP_NAME'])
    patient_metadata_selected_columns.loc[:, 'Gender_label'] = le.fit_transform(patient_metadata_selected_columns['GENDER'])
    data_with_patient_metadata = pd.merge(data_with_analyte_columns_selected, patient_metadata_selected_columns, on="PARENT_SAMPLE_NAME")
    # Reverse log transforming the data, so that gamma family can be applied in GLM model
    data_with_patient_metadata_reverse_log_transformed = reverse_log_transform(data_with_patient_metadata, analyte_columns_selected)
    #Removing patients (and their partners) with missing data
    house_to_exclude = data_with_patient_metadata_reverse_log_transformed[data_with_patient_metadata_reverse_log_transformed.isna().any(axis=1)].house.values
    data_with_patient_metadata_reverse_log_transformed_nan_removed = data_with_patient_metadata_reverse_log_transformed[~data_with_patient_metadata_reverse_log_transformed["house"].isin(house_to_exclude)]
    
    data_to_go = data_with_patient_metadata_reverse_log_transformed_nan_removed
    if data_type == "with_outlier":
        model_list = get_model_parallel(data_to_go, analyte_columns_selected, sample, False)
        model_filename = "GLM_global_compounds_with_outlier_sample_{}.joblib".format(sample)  
    elif data_type == "with_outlier_treated":
        data_to_go_treated_ = data_to_go[data_to_go.TREATMENT == "Treated"]
        house_to_include = data_to_go_treated_["house"].unique()
        data_to_go_treated = data_to_go[data_to_go.house.isin(house_to_include)]
        model_list = get_model_parallel(data_to_go_treated, analyte_columns_selected, sample, False)
        model_filename = "GLM_global_compounds_with_outlier_ms_treated_sample_{}.joblib".format(sample)
    elif data_type == "with_outlier_not_treated":
        data_to_go_not_treated_ = data_to_go[data_to_go.TREATMENT == "Off"]
        house_to_include = data_to_go_not_treated_["house"].unique()
        data_to_go_not_treated = data_to_go[data_to_go.house.isin(house_to_include)]
        model_list = get_model_parallel(data_to_go_not_treated, analyte_columns_selected, sample, False)
        model_filename = "GLM_global_compounds_with_outlier_ms_not_treated_sample_{}.joblib".format(sample)
    elif data_type == "without_outlier":
        model_list = get_model_parallel(data_to_go, analyte_columns_selected, sample, True)
        model_filename = "GLM_global_compounds_without_outlier_sample_{}.joblib".format(sample)
    elif data_type == "without_outlier_treated":
        data_to_go_treated_ = data_to_go[data_to_go.TREATMENT == "Treated"]
        house_to_include = data_to_go_treated_["house"].unique()
        data_to_go_treated = data_to_go[data_to_go.house.isin(house_to_include)]
        model_list = get_model_parallel(data_to_go_treated, analyte_columns_selected, sample, True)
        model_filename = "GLM_global_compounds_without_outlier_ms_treated_sample_{}.joblib".format(sample)
    elif data_type == "without_outlier_not_treated":
        data_to_go_not_treated_ = data_to_go[data_to_go.TREATMENT == "Off"]
        house_to_include = data_to_go_not_treated_["house"].unique()
        data_to_go_not_treated = data_to_go[data_to_go.house.isin(house_to_include)]
        model_list = get_model_parallel(data_to_go_not_treated, analyte_columns_selected, sample, True)
        model_filename = "GLM_global_compounds_without_outlier_ms_not_treated_sample_{}.joblib".format(sample)
            
    joblib.dump(model_list, os.path.join(OUTPUT_PATH, model_filename))
    print("Completed in {} hours".format(round((time.time()-start_time)/(60*60),2)))
    
    
if __name__ == "__main__":
    main()










