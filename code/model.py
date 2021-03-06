##########################################
###             model.py               ###
##########################################

# cd Documents/Project/PumpItUp_DrivenData/

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime


def top100cat(series):
    labels = series.copy()
    top100 = labels.value_counts()[:100].index.values
    notin100 = ~labels.isin(top100)
    labels[notin100] = 'minority'
    return labels

def top100cat_test(series_test, series_train):
    labels_test = series_test.copy()
    labels100 = set(series_train)
    notin100 = ~series_test.isin(labels100)
    labels_test[notin100] = 'minority'
    return labels_test

# def report_model_accuracy(model, X_train, y_train, X_test, y_test, other_acc=False):
#     clf = model.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     score = accuracy_score(y_pred=y_pred, y_true=y_test)
#     if not other_acc:
#         return(score)
#     else:
#         precision = precision_score(y_pred=y_pred, y_true=y_test, average="micro")
#         recall = recall_score(y_pred=y_pred, y_true=y_test, average="micro")
#         f1 = f1_score(y_pred=y_pred, y_true=y_test, average="micro")
#         return(score, precision, recall, f1)

# def ten_fold_crossvalidation(model, X, y, other_acc=False):
#     accuracies = list()
#     kf = KFold(n_splits=10)
#     for train, test in kf.split(X):
#         X_train = X[train]
#         y_train = y[train]
#         X_test = X[test]
#         y_test = y[test]
#         # print(X_train, y_train)
#         score = report_model_accuracy(model, X_train, y_train, X_test, y_test, other_acc)
#         print(score)
#         accuracies.append(score)
#     return(accuracies)

# def ave_report(result):
#     accuracy = np.mean([item[0] for item in result])
#     precision = np.mean([item[1] for item in result])
#     recall = np.mean([item[2] for item in result])
#     f1 = np.mean([item[3] for item in result])
#     return(round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3))


############# read data ##############
### train data
train_values = pd.read_csv('data/raw/Training_set_values.csv')
train_labels = pd.read_csv('data/raw/Training_set_labels.csv')
train_values = train_values.set_index('id')
train_labels = train_labels.set_index('id')
train_data = pd.concat([train_values, train_labels], axis=1, join='inner', join_axes=[train_labels.index])
train_data.sort_index(inplace=True)
del train_values, train_labels
### test data
test_values = pd.read_csv('data/raw/Test_set_values.csv')
test_values = test_values.set_index('id')
test_values.sort_index(inplace=True)

########### exploratory analysis ############
features = list(train_data.columns.values)

### label
train_data.status_group.value_counts()
# train_data.status_group.value_counts().plot.bar()  # barplot
label_code = LabelEncoder().fit(train_data.status_group)
train_data.status_group = label_code.transform(train_data.status_group)
test_values['status_group'] = np.NAN

### continuous variable flag
cont_var = ['date_recorded', 'gps_height', 'longitude', 'latitude', 'num_private', 'population', 'construction_year']

### 1. amount_tsh
sum(train_data.amount_tsh == 0) # too many 0s, discard
features.remove('amount_tsh')
### 2. date_record
train_data.date_recorded = train_data.date_recorded.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
d0 = datetime(2000,1,1,0,0,0)
train_data['date_recorded'] = train_data.date_recorded.apply(lambda x: abs((x-d0).days))
test_values.date_recorded = test_values.date_recorded.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
d0 = datetime(2000,1,1,0,0,0)
test_values.date_recorded = test_values.date_recorded.apply(lambda x: abs((x-d0).days))
### 3. funder
# sum(train_data.funder.isnull())
train_data.funder = top100cat(train_data.funder)  # convert low count value to minority
test_values.funder = top100cat_test(series_test=test_values.funder, series_train=train_data.funder)
### 4. gps_height
### 5. installer
train_data.installer.value_counts()
train_data.installer = top100cat(train_data.installer)
test_values.installer = top100cat_test(series_test=test_values.installer, series_train=train_data.installer)
### 6. longitude
# do not remove row, that will cause problem in test data
# train_data = train_data[train_data.longitude != 0] # remove longitude 0 rows
# test_values = test_values[test_values.longitude != 0]
### 7. latitude
# train_data.latitude.plot.density() # latitude is good
### 8. wpt_name
train_data.wpt_name.value_counts()
train_data.wpt_name = top100cat(train_data.wpt_name)
test_values.wpt_name = top100cat_test(series_test=test_values.wpt_name, series_train=train_data.wpt_name)
### 9. num_private
train_data.num_private.value_counts() # 56831 0s. (try to keep)
### 10. basin
train_data.basin.value_counts() # good feature
### 11. subvillage
train_data.subvillage.value_counts()
train_data.subvillage = top100cat(train_data.subvillage)
test_values.subvillage = top100cat_test(series_test=test_values.subvillage, series_train=train_data.subvillage)
### 12. region
train_data.region.value_counts() # region and region code are correlated features
features.remove('region')
### 13. region_code
train_data.region_code.value_counts() # good feature
train_data.region_code = train_data.region_code.apply(str) # convert to string
test_values.region_code = test_values.region_code.apply(str)
### 14. district_code
train_data.district_code.value_counts() # good feature
train_data.district_code = train_data.district_code.apply(str)
test_values.district_code = test_values.district_code.apply(str)
### 15. lga
train_data.lga.value_counts()
train_data.lga = top100cat(train_data.lga)
test_values.lga = top100cat_test(series_test=test_values.lga, series_train=train_data.lga)
### 16. ward
train_data.ward.value_counts()
train_data.ward = top100cat(train_data.ward)
test_values.ward = top100cat_test(series_test=test_values.ward, series_train=train_data.ward)
### 17. population
train_data.population.value_counts() # 19569 0s (try to keep)
### 18. public_meeting
# train_data = train_data[train_data.public_meeting.notnull()] # remove null row
# test_values = test_values[test_values.public_meeting.notnull()]
train_data.public_meeting[train_data.public_meeting.isnull()] = 'NA'
train_data.public_meeting = train_data.public_meeting.astype(str)
test_values.public_meeting = test_values.public_meeting.astype(str)
### 19. recorded_by
train_data.recorded_by.value_counts() # all same, no value
features.remove('recorded_by')
### 20. scheme_management
train_data.scheme_management.value_counts() # good features
### 21. scheme_name
sum(train_data.scheme_name.isnull()) # too many missing values
train_data.scheme_name = top100cat(train_data.scheme_name)
test_values.scheme_name = top100cat_test(series_test=test_values.scheme_name, series_train=train_data.scheme_name)
### 22. permit
# train_data = train_data[train_data.permit.notnull()]
# test_values = test_values[test_values.permit.notnull()]
train_data.permit = train_data.permit.astype(int)
test_values.permit = test_values.permit.astype(int)
### 23. construction_year
train_data.construction_year.value_counts() # too many missing values
### 24. extraction_type
train_data.extraction_type.value_counts() # good feature
### 25. extraction_type_group
features.remove('extraction_type_group') # duplicated feature
### 26. extraction_type_class
features.remove('extraction_type_class') # duplicated feature
### 27. management
train_data.management.value_counts() # good feature
### 28. management_group
features.remove('management_group') # duplicated feature
### 29. payment
train_data.payment.value_counts() # good feature
### 30. payment_type
features.remove('payment_type') # duplicated feature
### 31. water_quality
train_data.water_quality.value_counts() # good feature
### 32. quality_group
features.remove('quality_group') # duplicated feature
### 33. quantity
train_data.quantity.value_counts() # good feature
### 34. quantity_group
features.remove('quantity_group') # duplicated feature
### 35. source
train_data.source.value_counts() # good feature
### 36. source_type
features.remove('source_type') # duplicated feature
### 37. source_class
train_data.source_class.value_counts() # good feature
### 38. waterpoint_type
train_data.waterpoint_type.value_counts() # good feature
### 39. waterpoint_type_group
features.remove('waterpoint_type_group')

### remove unwanted features
train_data = train_data[features]
test_values = test_values[features]

### scale continuous variables
scaler = StandardScaler().fit(train_data[cont_var])
train_data[cont_var] = scaler.transform(train_data[cont_var])
test_values[cont_var] = scaler.transform(test_values[cont_var])

### clean characters
train_data.replace(r'[\s\/\\]+', '_', regex=True, inplace=True)
test_values.replace(r'[\s\/\\]+', '_', regex=True, inplace=True)
train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_values)
train_data_copy = train_data.copy()
train_data = train_data[test_data.columns]


###############  build model  ################
### split dataset as training and testing
train_label = train_data.status_group
train_feature = train_data.drop(labels='status_group', axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_label, test_size = 0.2, random_state = 1)
# X_train = train_feature.values
# y_train = train_label.values

### deicision tree
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
pred_tree = clf_tree.predict(X_test)
accuracy_score(y_true=y_test, y_pred=pred_tree)  # 0.758

### neural nework
clf_nn = MLPClassifier(hidden_layer_sizes=(50))
clf_nn.fit(X_train, y_train)
pred_nn = clf_nn.predict(X_test)
accuracy_score(y_true=y_test, y_pred=pred_nn)
# 20 0.777
# 20,10,5 0.765
# 40,20,10 0.770
# 50 0.776
# 100,50,25,10,5 0.767

### logitic regression
clf_logit = LogisticRegression(C=0.2)
clf_logit.fit(X_train, y_train)
pred_logit = clf_logit.predict(X_test)
accuracy_score(y_true=y_test, y_pred=pred_logit)
# 0.05 0.759
# 0.1 0.762
# 0.2 0.763
# 0.5 0.761
# 1 0.761

### random forest
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf.fit(X_train, y_train)
pred_rf = clf_rf.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=pred_rf))
# 10 0.794
# 25 0.801
# 100 0.805
# 200 0.808

# xgboost
clf_xgb = xgb.XGBClassifier()
clf_xgb.fit(X_train, y_train)
pred_xgb = clf_xgb.predict(X_test)
accuracy_score(y_true=y_test, y_pred=pred_xgb) # 0.760

############ result #############
res = pd.Series([0.758, 0.777, 0.763, 0.805, 0.760], index=['tree', 'nn', 'logit', 'rf', 'xgb'])


###########  prediction on test  #############
pred_rf_test = clf_rf.predict(test_data.drop(labels='status_group', axis=1))
test_data['status_group'] = label_code.inverse_transform(pred_rf_test)
### write output
submission = pd.read_csv('data/raw/Submission_format.csv', index_col=0)
for idx in submission.index:
    submission.loc[idx,'status_group'] = test_data.loc[idx, 'status_group']
### 0.8084 submitted

