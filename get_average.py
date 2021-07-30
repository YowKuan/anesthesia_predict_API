
import pandas as pd
import numpy as np
import pickle as pkl


with open('./processed_select_prean_2014_2020.pkl','rb') as f:
    dataa = pkl.load(f)
    serial_number = dataa['病歷編號']
    dataa = dataa[['性別', '身高', '體重', '年齡', '病史_DM', '病史_HYPERLIPIDEMIA',
    '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE', '病史_COPD',
    '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
    '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING',
    '病史_ALLERGY', 'LABDATA_HB', 'LABDATA_PLATELET',
    'LABDATA_INR', 'LABDATA_PT', 'LABDATA_APTT',
    'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 'LABDATA_SUGAR',
    'LABDATA_NA', 'LABDATA_K', 
    '麻醉危險分級(ASA Class)',
    'ASA_CLASS_E',
    'BT', 'SPO2', 'HR', 'RR', '意識',
    'SBP', 'DBP']]

    dataa.rename(columns={'麻醉危險分級(ASA Class)':'麻醉危險分級_ASA_CLASS'},inplace=True)

average = dataa.mean()
with open('averaged_select_prean_2014_2020.pkl','wb') as f:
    pkl.dump(average,f)