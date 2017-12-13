

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from numpy import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#read data from csv
train_values = pd.read_csv('data/raw/Training_set_values.csv')
train_labels = pd.read_csv('data/raw/Training_set_labels.csv')
test_values = pd.read_csv('data/raw/Test_set_values.csv')
train_data = pd.concat([train_values,test_values])


#drop redudant features
train_data.drop(['recorded_by'], axis=1, inplace=True)
train_data.drop(['region'], axis=1, inplace=True)
train_data.drop(['extraction_type','extraction_type_group'], axis=1, inplace=True)
train_data.drop(['management'], axis=1, inplace=True)
train_data.drop(['payment'], axis=1, inplace=True)
train_data.drop(['water_quality'], axis=1, inplace=True)
train_data.drop(['quantity'], axis=1, inplace=True)
train_data.drop(['source','source_class'],axis=1,inplace=True)
train_data.drop(['waterpoint_type'], axis=1, inplace=True)


#change train_label values to int
train_labels['status_group'] = train_labels['status_group'].astype('category')
train_labels['status_group'] = train_labels['status_group'].cat.codes


#define functions to transfrom data to binary and category
def binary(x):
    if (x == True) or (x == 'True'):
        return 1
    elif (x == False) or (x == 'False'):
        return 0
def category(x):
    if (x == 0):
        return 'functional'
    elif (x == 1):
        return 'functional needs repair'
    elif (x == 2):
        return 'non functional'


#apply binary function and fill na with -1
train_data['public_meeting'] = train_data['public_meeting'].apply(binary)
train_data['public_meeting'].fillna(-1, inplace=True)


#apply binary function and fill na with -1
train_data['permit'] = train_data['permit'].apply(binary)
train_data['permit'].fillna(-1, inplace=True)


#change date_recorded to year
train_data.date_recorded = train_data.date_recorded.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
train_data['recorded_year'] = train_data['date_recorded'].dt.year

#define operational_years as recorded_year - construction_year
op_years = list(train_data.recorded_year-train_data.construction_year)
operational_years = []
for i in op_years:
    if (i > 500) or (i < 0):
        operational_years.append(0)
    else:
        operational_years.append(i)

train_data['operational_years'] = operational_years


#drop redundant features
train_data.drop(['recorded_year'], axis=1, inplace=True)
train_data.drop(['construction_year'], axis=1, inplace=True)
train_data.drop(['date_recorded'], axis=1, inplace=True)


#change waterpoint_type_group to category then to integer
train_data['waterpoint_type_group'] = train_data['waterpoint_type_group'].astype('category')
train_data['waterpoint_type_group'] = train_data['waterpoint_type_group'].cat.codes


#change source_type to category then to integer
train_data['source_type'] = train_data['source_type'].astype('category')
train_data['source_type'] = train_data['source_type'].cat.codes


#change quantity_group to category then to integer
train_data['quantity_group'] = train_data['quantity_group'].astype('category')
train_data['quantity_group'] = train_data['quantity_group'].cat.codes


#change quantity_group to category then to integer
train_data['quality_group'] = train_data['quality_group'].astype('category')
train_data['quality_group'] = train_data['quality_group'].cat.codes


#change payment_type to category then to integer
train_data['payment_type'] = train_data['payment_type'].astype('category')
train_data['payment_type'] = train_data['payment_type'].cat.codes


#change management_group to category then to integer
train_data['management_group'] = train_data['management_group'].astype('category')
train_data['management_group'] = train_data['management_group'].cat.codes


#change extraction_type_class to category then to integer
train_data['extraction_type_class'] = train_data['extraction_type_class'].astype('category')
train_data['extraction_type_class'] = train_data['extraction_type_class'].cat.codes


#change basin to category then to integer
train_data['basin'] = train_data['basin'].astype('category')
train_data['basin'] = train_data['basin'].cat.codes


#change scheme_management to category then to integer
train_data['scheme_management'] = train_data['scheme_management'].astype('category')
train_data['scheme_management'] = train_data['scheme_management'].cat.codes


#change funder to category then to integer
train_data['funder'] = train_data['funder'].astype('category')
train_data['funder'] = train_data['funder'].cat.codes


#change installer to category then to integer
train_data['installer'] = train_data['installer'].astype('category')
train_data['installer'] = train_data['installer'].cat.codes


#change wpt_name to category then to integer
train_data['wpt_name'] = train_data['wpt_name'].astype('category')
train_data['wpt_name'] = train_data['wpt_name'].cat.codes


#change lga to category then to integer
train_data['lga'] = train_data['lga'].astype('category')
train_data['lga'] = train_data['lga'].cat.codes


#change ward to category then to integer
train_data['ward'] = train_data['ward'].astype('category')
train_data['ward'] = train_data['ward'].cat.codes


#change scheme_name to category then to integer
train_data['scheme_name'] = train_data['scheme_name'].astype('category')
train_data['scheme_name'] = train_data['scheme_name'].cat.codes


#change subvillage to category then to integer
train_data['subvillage'] = train_data['subvillage'].astype('category')
train_data['subvillage'] = train_data['subvillage'].cat.codes

#drop na values
train_data = train_data.dropna()


#use random forest classifier
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100)

#split data
train_label = train_labels['status_group']
train_feature, test_data = train_data[:59400], train_data[59400:]


X_train, X_test, y_train, y_test = train_test_split(train_feature, train_label, test_size = 0.2, random_state = 1)


rf.fit(train_feature, train_label)


prediction = rf.predict(test_data)

#generate submission dataframe then output to csv
submission = pd.DataFrame({ 'id': test_data['id'],
                            'status_group': prediction })
submission['status_group'] = submission['status_group'].apply(category)

submission.to_csv("submission.csv", index=False)


#use xgboost classifier
import xgboost as xgb


xgbt = xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.05)

xgbt.fit(X_train, y_train)

xgb_prediction = xgbt.predict(X_test)

accuracy_score(y_true=y_test, y_pred = xgb_prediction)


#use neural_network
from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(solver='lbfgs', activation='relu',
                    hidden_layer_sizes=(50, 30), max_iter=10000, random_state=1)


mlp.fit(X_train, y_train)

mlp_prediction = mlp.predict(X_test)

accuracy_score(y_true=y_test, y_pred = mlp_prediction)


#use svm classifier
from sklearn import svm


svc = svm.SVC(gamma=0.001, C=10, kernel='poly')

svc.fit(X_train, y_train)

svc_prediction = svc.predict(X_test)

accuracy_score(y_true=y_test, y_pred = svc_prediction)

