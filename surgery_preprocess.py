import pandas as pd
import numpy as np
import pickle as pkl
import math
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
if not os.path.exists('./model'):
    os.makedirs('./model')
if not os.path.exists('./ConfusionMatrix'):
    os.makedirs('./ConfusionMatrix')


def load_data():

    with open('./converted_csv.csv','rb') as f:
        #data = pkl.load(f)
        data = pd.read_csv(f, encoding = 'big5')
    data['性別'] = data['性別'].apply(lambda x: preprocessing.sex(x))
    data['年齡'] = data['年齡'].fillna(np.nan).apply(preprocessing.age)
    data['體重'] = data['體重'].astype(str).fillna(np.nan).apply(preprocessing.weight)
    data['身高'] = data['身高'].astype(str).fillna(np.nan).apply(preprocessing.height)
    data[['病史_DM','病史_HYPERLIPIDEMIA', '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE',
                '病史_COPD', '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
                '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING','病史_ALLERGY']] = data[['病史_DM','病史_HYPERLIPIDEMIA', '病史_HYPERTENSION', '病史_CVA', '病史_CARDIAC_DISEASE',
                '病史_COPD', '病史_ASTHMA', '病史_HEPATIC_DISEASE', '病史_RENAL_DISEASE',
                '病史_BLEEDING_DISORDER', '病史_MAJOR_OPERATIONS', '病史_SMOKING','病史_ALLERGY']].applymap(preprocessing.history)

    data[['LABDATA_HB', 'LABDATA_PLATELET', 'LABDATA_INR', 'LABDATA_PT', 
            'LABDATA_APTT', 'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 
            'LABDATA_SUGAR', 'LABDATA_NA', 'LABDATA_K']] = data[['LABDATA_HB', 
                                                                'LABDATA_PLATELET', 'LABDATA_INR', 
                                                                'LABDATA_PT', 'LABDATA_APTT',
                                                                'LABDATA_CR', 'LABDATA_GOT', 'LABDATA_GPT', 
                                                                'LABDATA_SUGAR', 'LABDATA_NA', 'LABDATA_K']].astype(str).fillna(np.nan).applymap(preprocessing.lab)

    data['ASA_CLASS_E'] = data['ASA_CLASS_E'].astype(str).fillna(np.nan).apply(preprocessing.ASA_CLASS_E)
    data['BT'] = data['BT'].astype(str).apply(preprocessing.strQ2B).apply(preprocessing.BT)
    data['SPO2'] = data['SPO2'].astype(str).apply(preprocessing.strQ2B).apply(preprocessing.spo2)
    data['HR'] = data['HR'].astype(str).apply(preprocessing.strQ2B).fillna(np.nan).apply(preprocessing.HR)
    data['RR'] = data['RR'].astype(str).apply(preprocessing.strQ2B).fillna(np.nan).apply(preprocessing.RR)
    #data['BP'] = data['BP'].astype(str).apply(preprocessing.strQ2B).fillna(np.nan).apply(lambda x: preprocessing.BP(x))#, axis=1, result_type='expand')
    #data['SBP'] = data['BP'].apply(lambda x: preprocessing.a(x))
    #data['DBP'] = data['BP'].apply(lambda x: preprocessing.b(x))
    data['意識'] = data['意識'].astype(str).apply(preprocessing.strQ2B).fillna(np.nan).apply(preprocessing.consciousness)

    sex_bg = {'男': 1, '女': 0}
    consense = {'清楚': 1, 'nan':0}
    tmp = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'None':0}
    data['性別'] = data['性別'].replace(sex_bg)
    data['意識'] = data['意識'].apply(str).replace(consense).fillna(0)
    data['麻醉危險分級_ASA_CLASS'] = data['麻醉危險分級_ASA_CLASS'].replace(tmp)
    #data['麻醉危險分級(ASA Class)'] = data['麻醉危險分級(ASA Class)'].replace(tmp)

    with open('patient_data.pkl','wb') as f:
        pkl.dump(data,f)

class preprocessing():
    def sex(x):
        if x not in ['男','女']:
            return np.nan
        return x

    def age(x):
        try:
            return float(x)
        except:
            return x


    def weight(x):
        x = x.strip('\' +-=~<>約\/`？a*')
        try:
            return float(x)
        except:
            if '?' in x:
                return(np.nan)
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'None','不詳']:
                return(np.nan)
            elif 'g' in x:
                x = x.strip('gms')
                return(float(x)/1000)
            elif 'X' in x:
                x = x.strip('X')
                return(float(x*10))
            elif 'x' in x:
                x = x.strip('x')
                return(float(x*10))
            else:
                try:
                    return(float(x[0:2]))
                except:
                    print(x,'Wrong Format')


    def height(x):
        x = x.strip('\' +-=~<>約\/`？a*')
        try:
            return float(x)
        except:
            if '?' in x:
                return(np.nan)
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'None','不詳','未測']:
                return(np.nan)
            elif 'X' in x:
                x = x.strip('X')
                return(float(x*10))
            elif 'x' in x:
                x = x.strip('x')
                return(float(x*10))
            else:
                try:
                    return(float(x[0:2]))
                except:
                    print(x,'Wrong Format')


    def history(x):
        try:
            if x == None:
                return x
            elif 'yes' in x:
                return True
            elif 'denied' in x:
                return False
        except:
            print(x, 'Wrong Format')

    def lab(x):
        x = x.strip('\' +-=~<>約\/`？a*')
        try:
            return float(x)
        except:
            if x == 'None':
                return np.nan
            elif '?' in x:
                return np.nan
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'None','不詳','未測']:
                return np.nan
            elif 'X' in x:
                x = x.strip('X')
                return float(x*10)
            elif 'x' in x:
                x = x.strip('x')
                return float(x*10)
    #         else:
    #             try:
    #                 return float(x[0:2])
    #             except:
    #                 print(x,'Wrong Format')

    def ASA_CLASS_E(x):
        if 'E' in x:
            return True
        else:
            return False

    def strQ2B(ustring):
        """把字串全形轉半形"""
        rstring = ""
        for uchar in ustring:
            inside_code=ord(uchar)
            if inside_code==0x3000:
                inside_code=0x0020
            else:
                inside_code-=0xfee0
                if inside_code<0x0020 or inside_code>0x7e:   #轉完之後不是半形字元返回原來的字元
                    rstring += uchar
        return rstring
    def BT(x):
    #     x = x.strip('\' +-=~<>約\/`？a*.無牙DGQqRAMBULmbu/ˇˊ，。:%\’)Cc').replace(',','.').replace(';','').replace(' ','').replace('..','.') .strip('\=.ˇˊ')
        x = x.strip(' QqDG無牙-\=.ˇˊ`').replace('?','').replace(',','.').replace(';','').replace(' ','').replace('..','.')
    #     print(x,end='\r')
        try:
            return float(x)
        except:
            if x == 'None':
                return np.nan
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'none','不詳','未測', '測量不到', '-', '---', '。','’']:
                return np.nan
            elif 'X' in x:
                x = x.strip('X')
                return float(x*10)
            elif 'x' in x:
                x = x.strip('x')
                return float(x*10)
            elif x in ['3.6.3', '3.7.5', '3.6.7', '3.6.6']:
                return float(x.replace('.','',1))
            else:
                print(x,'Wrong Format')
                return np.nan
                
    # def BT2(x):
    #     m = re.search(r'(\d|\.)*/(\d|\.)*',x)
    #     m =  m.group()
    #     return float(m)
        



    def spo2(x):
        x = str(x)
        if x == 'None':
            return np.nan
        x = x.strip('-`+Lq%*/VE').replace(',','.').replace(';','').replace(' ','').replace('..','.').replace('AMBU','')
        x = x.replace('ambu','').replace('N/C','').replace('n/c','').replace('RA','').replace(':','').replace('()','').replace('3L','')
        x = x.replace('O2USE','').replace('masl6L','').replace('/MIN','')
        try:
            y = float(x)
            if 1500 > y >= 1000:
    #             print(x,y/10)
                return y/10
            elif 15000 > y >= 1500:
    #             print(x,y/100)
                return y/100
            elif 100000 > y >= 15000:
                return y/1000
            else:
                return float(x)
        except:
            if '?' in x:
                return np.nan
            
            elif '*' in x:
                return np.nan
            
            elif '+' in x:
                return np.nan
            
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            
            elif x in ['', 'none','不詳','未測', '測量不到']:
                return np.nan
            
            elif 'X' in x:
                x = x.strip('X')
                return float(x*10)
            elif 'x' in x:
                x = x.strip('x')
                return float(x*10)
            
            else:
    #             print(x)
                return np.nan

    def HR(x):
        x = x.strip('*-/qAf`').replace(',','.').replace(';','').replace(' ','').replace('..','.')
    #     print(x+'         ',end='\r')
        try:
            return float(x)
        except:
            if x == 'None':
                return np.nan
            elif '?' in x:
                return np.nan
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'none','不詳','未測', '測量不到','-','=']:
                return np.nan
            elif 'X' in x:
                x = x.strip('X')
                return float(x*10)
            elif 'x' in x:
                x = x.strip('x')
                return float(x*10)
            elif '/' in x:
                return np.nan
            elif '==' in x:
                return np.nan
            else:
                print(x,'Wrong Format')
                return np.nan


    def RR(x):
        print(x+'         ',end='\r')
        x = x.strip('`+wmEhi呼吸器擠壓').replace('ventilator','').replace('Ventilator','').replace('bambu','').replace('AMBU','').replace('ambu','').replace('Ambu','').replace('amb','').replace('abu','')
        x = x.replace('o2','').replace('O2','').replace('3L','').replace('use','').replace('AR','').replace('MV/','')
        x = x.replace('/VT','').replace('/TV','').replace('tv','').replace('vt','').replace('V/','').replace('v/','').replace('/v','')
        x = x.replace('/V','').replace('/T','').replace('/t','').replace('VT','').replace('vt','').replace('V','')
        x = x.replace('N/C','').replace('n/c','').replace('/MV','').replace('MV','').replace('/mv','')
        x = x.replace('`','').replace(',','.').replace(';','').replace(' ','').replace('..','.').strip('AMCR/:-')
        

        try:
            return float(x)
        except:
            if x == 'None':
                return np.nan
            elif '?' in x:
                return np.nan
            elif 'L' in x:
                return np.nan
            elif '/' in x:
                return np.nan
            elif '-' in x:
                x = x.split('-')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif '~' in x:
                x = x.split('~')
                a, b = float(x[0]), float(x[1])
                return max([a, b])
            elif x in ['', 'none','不詳','未測', '測量不到', '呼吸器','endo','AC']:
                return np.nan
            elif 'x' in x:
                x = x.strip('x')
                return float(x*10)
            elif '-' in x:
                x = x.strip('%').split('-')
                
                #print(x[0],x[1])
                return max(float(x[0],float(x[1])))
            else:
                #print('\n')
                #print(x,'Wrong Format')
                return np.nan

    # data['RR'].unique()#.astype(str).apply(strQ2B).fillna(np.nan).apply(RR)



    def BP(x):
        x = x.replace('---', 'None').replace('=7', '').replace('//','/').replace('\\','/').strip('()*- `無法監量側未測不到左手左腳RHmeanilILoOMEANJsr')
        if x in['None','NIL','-/-','-', '--', '?','?/?', 'nil','Nil','未量','上述', '未側','未測量','測量不到','未測/未測','','NoneNone','NoneNone-','NoneNone--','/', 'no','NO','No','N/M','N/A','       /','--/--','None-','None--','無法監測','0','未','x']:
            return [np.nan, np.nan]
    #     if x in list1:
    #         return [float(x), float(x)]
    #     elif x in list2:
    #         x = x.split('/')
    #         return [float(x[0]), np.nan]
        if x in ['155/*77','130*71','125*/85']:
            x = x.replace('*','/').replace('//','/')
            x = x.split('/')
            return [float(x[0]), float(x[1])]
        elif '?' in x:
            return [np.nan, np.nan]
        elif x == '37.2':
            return [np.nan, np.nan]
        else:
            
            x = x.split('/')
            try:
                x = [abs(float(x[0])), abs(float(x[1]))]
            except:
                
                x = [abs(float(x[0][0:math.ceil(len(x[0])/2)])), abs(float(x[0][math.ceil(len(x[0])/2):6]))]
            
            return x
    
    def a(x):
        return x[0]
    def b(x):
        return x[1]


    def consciousness(x):
        if '清楚' in x or '活動' in x:
            return '清楚'
        else:
            return np.nan





