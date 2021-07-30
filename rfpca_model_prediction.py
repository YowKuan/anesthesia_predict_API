import pandas as pd
import numpy as np
import pickle as pkl
import re
import glob
import datetime
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import normalize
from sklearn.utils.random import sample_without_replacement
import joblib
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.impute import KNNImputer
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import json
import surgery_preprocess
if not os.path.exists('./model'):
    os.makedirs('./model')
import shap




def prediction():

    with open('./averaged_select_prean_2014_2020.pkl','rb') as g:
        dataa = pkl.load(g)
        print(dataa)
        #serial_number = dataa['病歷編號']
        dataa = dataa[['性別', '身高', '體重', '年齡', '病史_DM', '病史_HYPERLIPIDEMIA',
        '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE', '病史_COPD',
        '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
        '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING',
        '病史_ALLERGY', 'LABDATA_HB', 'LABDATA_PLATELET',
        'LABDATA_INR', 'LABDATA_PT', 'LABDATA_APTT',
        'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 'LABDATA_SUGAR',
        'LABDATA_NA', 'LABDATA_K', 
        '麻醉危險分級_ASA_CLASS',
        'ASA_CLASS_E',
        'BT', 'SPO2', 'HR', 'RR', '意識',
        'SBP', 'DBP']]

    with open('./patient_data.pkl','rb') as f:
        data = pkl.load(f)
        serial_number = data['病歷編號']
        data = data[['性別', '身高', '體重', '年齡', '病史_DM', '病史_HYPERLIPIDEMIA',
        '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE', '病史_COPD',
        '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
        '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING',
        '病史_ALLERGY', 'LABDATA_HB', 'LABDATA_PLATELET',
        'LABDATA_INR', 'LABDATA_PT', 'LABDATA_APTT',
        'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 'LABDATA_SUGAR',
        'LABDATA_NA', 'LABDATA_K', 
        '麻醉危險分級_ASA_CLASS',
        'ASA_CLASS_E',
        'BT', 'SPO2', 'HR', 'RR', '意識',
        'SBP', 'DBP']]

    data[['身高', '體重', '年齡','LABDATA_HB', 'LABDATA_PLATELET',
        'LABDATA_INR', 'LABDATA_PT', 'LABDATA_APTT',
        'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 'LABDATA_SUGAR',
        'LABDATA_NA', 'LABDATA_K','BT', 'SPO2', 'HR', 'RR','SBP', 'DBP']] = data[['身高', '體重', '年齡','LABDATA_HB', 'LABDATA_PLATELET',
        'LABDATA_INR', 'LABDATA_PT', 'LABDATA_APTT',
        'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 'LABDATA_SUGAR',
        'LABDATA_NA', 'LABDATA_K','BT', 'SPO2', 'HR', 'RR','SBP', 'DBP']].fillna(dataa)

    data[['性別','病史_DM', '病史_HYPERLIPIDEMIA',
        '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE', '病史_COPD',
        '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
        '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING',
        '病史_ALLERGY',
        '麻醉危險分級_ASA_CLASS',
        'ASA_CLASS_E','意識']]=data[['性別','病史_DM', '病史_HYPERLIPIDEMIA',
        '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE', '病史_COPD',
        '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
        '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING',
        '病史_ALLERGY',
        '麻醉危險分級_ASA_CLASS', 'ASA_CLASS_E','意識']].fillna(dataa)

    X = data
    #print(X)
    #Y = output

    rfpca_model = joblib.load('./model/RandomForest_with_pca.pkl')

    ####training part - not required now
    #np.random.seed(999)
    #train_id, test_id, _, _ = train_test_split(serial_number.unique(), serial_number.unique(), shuffle = True, test_size = 0.25)
    #train = serial_number.apply(lambda x: x in train_id)
    #test = (train*-1+1).astype(bool)
    #x_train = X.loc[train]
    #x_test = X.loc[test]

    prediction_result = rfpca_model.predict_proba(X)
    to_json = prediction_result.tolist()
    print(prediction_result)

    ##This part is the SHAP explainable AI part.
    # Use TreeExplainer to create object that calculate shap values
    explainer = shap.TreeExplainer(rfpca_model)

    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    value_list = data.values.tolist()
    value_list = value_list[0]
    shap_list = shap_values[0][0]
    positive = []
    negative = []
    for (column, value, shap_val) in zip(data.columns, value_list, shap_list):
        if shap_val > 0:
            positive.append([column, value, shap_val])
        else:
            negative.append([column, value, shap_val])
    #print(positive)
    def normalize(input_list, reverse):       
        s = 0  
        input_list = sorted(input_list, key=lambda x: x[2], reverse=reverse)
        for column, value, shap_val in input_list:
            s += shap_val
        mult = abs(100/s)
        for i in range(len(input_list)):
            input_list[i][2] *= mult
        return list(reversed(input_list))[:10]
    positive = normalize(positive,  False)
    negative = normalize(negative, True)

    
    pre_result = {'Result': to_json, 'Positive_cause': positive, 'Negative_cause': negative}

    with open('predict_result.json', 'w') as outfile:
        json.dump(pre_result, outfile)

