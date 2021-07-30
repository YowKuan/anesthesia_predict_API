#from treeinterpreter import treeinterpreter as ti
import joblib
import pandas as pd
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor

rfpca_model = joblib.load('./model/RandomForest_with_pca.pkl')

with open('./averaged_select_prean_2014_2020.pkl','rb') as g:
    dataa = pkl.load(g)
    #print(dataa)
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
    print(dataa)
    dataa.to_csv("averaged_select_prean_2014_2020.csv")

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
X.to_csv('patient_data.csv')
#fit a scikit-learn's regressor model
# rf = RandomForestRegressor()
# rf.fit(trainX, trainY)

# prediction, bias, contributions = ti.predict(rfpca_model, X)
# print("prediction:", prediction)
# print("bias:", bias)
# print("contributions:", contributions[0])

# for i in range(len(contributions[0])):
#     print(contributions[0][i][0])

import shap  # package used to calculate Shap values

# Use TreeExplainer to create object that calculate shap values
# you can use other to fit other kind of model
# shap.DeepExplainer works with Deep Learning models.
# shap.KernelExplainer works with all models, though it is slower than other Explainers 
# and it offers an approximation rather than exact Shap values.
explainer = shap.TreeExplainer(rfpca_model)

# Calculate Shap values
shap_values = explainer.shap_values(X)
print(shap_values)
#shap.plots.bar(shap_values)
# shap.initjs()
# shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)