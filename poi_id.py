#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import operator
import scikitplot as skplt
import seaborn as sns; sns.set(style="ticks", color_codes=True)


import pprint
pretty = pprint.PrettyPrinter()



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Convert the file to Pandas dataframe
    
enron = pd.DataFrame.from_dict(data_dict, orient = 'index')

print 'Example Value Dictionary of Features'
pretty.pprint(data_dict['ALLEN PHILLIP K']) 
pretty.pprint(len(data_dict['ALLEN PHILLIP K'])) 



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Leaving email address from the original list of features out, 
#as that is not something measureable or countable, and the information I could 
#get from it is redundant to the key value.

features_list = [
        'poi',
        'salary',
        'deferral_payments',
        'total_payments',
        'loan_advances',
        'bonus',
        'restricted_stock_deferred',
        'deferred_income',
        'total_stock_value',
        'expenses',
        'exercised_stock_options',
        'other',
        'long_term_incentive',
        'restricted_stock',
        'director_fees', 
        'to_messages',
        'from_poi_to_this_person',
        'from_messages',
        'from_this_person_to_poi',
        'shared_receipt_with_poi'] 



names = sorted(data_dict.keys())  #sort names of Enron employees in dataset by first letter of last name

print 'Sorted list of Enron employees by last name'
pretty.pprint(names) 



### Total Number of Data Points
print 'Total Number of data points: %d' %len(data_dict)



### Number of POIs
num_poi = len(enron[enron['poi'].astype(np.float32)==1])
num_non_poi = len(data_dict) - num_poi

print 'Number of POIs:' + str(num_poi)
print 'Number of Non-POIs:' + str(num_non_poi)



### Number of Features
print 'Number of features: %d' %len(features_list)



### Missing Features
from tabulate import tabulate

features_nan = {}
for name, feature in data_dict.iteritems():
    for feature_key, val in feature.iteritems():
        if val == 'NaN':
            # Assign 0 to value
            feature[feature_key] = 0
            if features_nan.get(feature_key):
                features_nan[feature_key] = features_nan.get(feature_key) + 1
            else:
                features_nan[feature_key] = 1

print '# of Missing Values by Feature:'
print ''
print ("{:<25} {:<5}".format('FEATURE', 'COUNT'))
print '-------------------------------'
for key, value in features_nan.items(): 
    print ("{:<25} {:<5}".format(key, value))





### Task 2: Remove outliers

features = ['salary', 'bonus', 'poi']
data = featureFormat(data_dict, features, 'poi')
plt.figure(figsize=(20, 10))
plt.xlabel('salary')
plt.ylabel('bonus')
for point in data:
    salary = point[0]
    bonus = point[1]
    poi = point[2]
    if poi:
        color = 'red'
    else:
        color = 'blue'
    plt.scatter( salary, bonus, color = color )



#Remove outlier "Total", which is just the summary/total line from the spreadsheet. 
data_dict.pop('TOTAL', 0)



#Remove outlier "LOCKHART EUGENE E", for whom all data is NaN or 0. 
data_dict.pop('LOCKHART EUGENE E', 0)



features = ['salary', 'bonus', 'poi']
data = featureFormat(data_dict, features, 'poi')
plt.figure(figsize=(20, 10))
plt.xlabel('salary')
plt.ylabel('bonus')
plt.legend()
for point in data:
    salary = point[0]
    bonus = point[1]
    poi = point[2]
    if poi:
        color = 'red'
    else:
        color = 'blue'
    plt.scatter( salary, bonus, color = color )



#Create pairsplot to get an overview of what interesting relationships there may be 
#between features that could be used to engineer new features.

g = sns.pairplot(enron, vars=['bonus','exercised_stock_options','from_messages','from_poi_to_this_person', 'from_this_person_to_poi'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['o','+'])



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


def computeRatio(messages, features_list):
    ratio = 0.
    if (messages == 0 or features_list == 0):
        return ratio
    ratio = messages / float(features_list)
    return ratio


def newFeatures(my_dataset):
    for poi_name in my_dataset:
        data_point = my_dataset[poi_name]
        data_point['from_poi_to_this_person_ratio'] = computeRatio(data_point['from_poi_to_this_person'],
                                                                   data_point['to_messages'])
        data_point['from_this_person_to_poi_ratio'] = computeRatio(data_point['from_this_person_to_poi'],
                                                                   data_point['from_messages'])
        data_point['bonus_salary_ratio'] = computeRatio(data_point['bonus'],
                                                                   data_point['salary'])
        data_point['bonus_total_payments_ratio'] = computeRatio(data_point['bonus'],
                                                                   data_point['total_payments'])
        data_point['shared_receipt_with_poi_to_messages_ratio'] = computeRatio(data_point['shared_receipt_with_poi'],
                                                                   data_point['to_messages'])

    return my_dataset, ['from_poi_to_this_person_ratio', 
                        'from_this_person_to_poi_ratio', 
                        'bonus_salary_ratio', 
                        'bonus_total_payments_ratio',
                        'shared_receipt_with_poi_to_messages_ratio' 
                       ]

my_dataset, new_features = newFeatures(my_dataset)
features_list = features_list + new_features

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


pretty.pprint (features_list)



features_list_ratios = ['poi', 'from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']    
    ### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list_ratios
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list_ratios)

### plot new features
plt.figure(figsize=(20, 10))
for point in data:
    if point[0] == 1:
        color = 'red'
    else:
        color = 'blue'
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi, color = color )
plt.xlabel('% of Emails from POI')
plt.ylabel('% of Emails to POI')
plt.show()



enron_updated = pd.DataFrame.from_dict(my_dataset, orient = 'index')

print 'Example Value Dictionary with New Features'
pretty.pprint(my_dataset['ALLEN PHILLIP K']) 
pretty.pprint(len(my_dataset['ALLEN PHILLIP K'])) 



#Create new pairsplot to get an overview of what interesting relationships there may be 
#between features including new features.

g = sns.pairplot(enron_updated, vars=['salary', 
                                      'deferral_payments', 
                                      'total_payments', 
                                      'bonus', 
                                      'deferred_income', 
                                      'total_stock_value', 
                                      'expenses', 
                                      'exercised_stock_options', 
                                      'long_term_incentive', 
                                      'director_fees', 
                                      'shared_receipt_with_poi', 
                                      'from_poi_to_this_person_ratio', 
                                      'from_this_person_to_poi_ratio',
                                      'bonus_salary_ratio', 
                                      'bonus_total_payments_ratio',
                                      'shared_receipt_with_poi_to_messages_ratio'
                                     ],
                 dropna=True, diag_kind='kde', hue='poi', markers=['o','+'])




#Just the ratios

g = sns.pairplot(enron_updated, vars=['from_poi_to_this_person_ratio', 
                                      'from_this_person_to_poi_ratio',
                                      'bonus_salary_ratio', 
                                      'bonus_total_payments_ratio',
                                      'shared_receipt_with_poi_to_messages_ratio'
                                     ],
                 dropna=True, diag_kind='kde', hue='poi', markers=['o','+'])



# function using SelectKBest
def findKbestFeatures(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    # print "sorted_pairs", sorted_pairs
    k_best_features = dict(sorted_pairs[:k])

    return k_best_features

num_features = 15
selectedBestFeatures = findKbestFeatures(my_dataset, features_list, num_features)
sortedBestFeatures = sorted(selectedBestFeatures.items(), key=lambda x: x[1], reverse = True)
selectedFeatures = ['poi'] + selectedBestFeatures.keys()


print "SELECTED 15 BEST FEATURES BY KBEST:"
print ''
print ("{:<30} {:<5}".format('FEATURE', 'STRENGTH'))
print '--------------------------------------------'
for i in sortedBestFeatures: 
    print ("{:<30} {:<5}".format(i[0], i[1]))



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.

features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]


print 'Decision Tree Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)
print '\nFeature Ranking: '
for i in range(24):
    print "{} feature {} ({})".format(i+1,features_list[i+1],round(importances[indices[i]],4))

skplt.metrics.plot_confusion_matrix(labels_test, prediction, normalize=True)



clf = GaussianNB()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)


print 'Gaussian Naive Bayes Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)

skplt.metrics.plot_confusion_matrix(labels_test, prediction, normalize=True)



clf = SVC()

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'Support Vector Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



clf = KNeighborsClassifier()

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'K-Neighbors Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



clf = RandomForestClassifier()

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'Random Forest Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



clf = AdaBoostClassifier()

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'AdaBoost Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



clf = AdaBoostClassifier(n_estimators = 1000, learning_rate = 1.0, algorithm = 'SAMME')

clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'AdaBoost Classifier - Partially Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)

skplt.metrics.plot_confusion_matrix(labels_test, prediction, normalize=True)




features_list_for_lr = ["bonus", "salary"]
data = featureFormat( data_dict, features_list_for_lr, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train,target_train)

print 'Coefficient: ', reg.coef_[0]
print 'Intercept: ', reg.intercept_
print ''
print 'Stats on Training dataset'
print 'r-squared score: ', round(reg.score(feature_train,target_train), 3)
print ''
print 'Stats on Test dataset'
print 'r-squared score: ', round(reg.score(feature_test,target_test), 3)



features_list_for_lr = ["exercised_stock_options", "bonus"]
data = featureFormat( data_dict, features_list_for_lr, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train,target_train)

print 'Coefficient: ', reg.coef_[0]
print 'Intercept: ', reg.intercept_
print ''
print 'Stats on Training dataset'
print 'r-squared score: ', round(reg.score(feature_train,target_train), 3)
print ''
print 'Stats on Test dataset'
print 'r-squared score: ', round(reg.score(feature_test,target_test), 3)



features_list_for_lr = ["exercised_stock_options", "long_term_incentive"]
data = featureFormat( data_dict, features_list_for_lr, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train,target_train)

print 'Coefficient: ', reg.coef_[0]
print 'Intercept: ', reg.intercept_
print ''
print 'Stats on Training dataset'
print 'r-squared score: ', round(reg.score(feature_train,target_train), 3)
print ''
print 'Stats on Test dataset'
print 'r-squared score: ', round(reg.score(feature_test,target_test), 3)



features_list_for_lr = ["bonus", "long_term_incentive"]
data = featureFormat( data_dict, features_list_for_lr, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg.fit(feature_train,target_train)

print 'Coefficient: ', reg.coef_[0]
print 'Intercept: ', reg.intercept_
print ''
print 'Stats on Training dataset'
print 'r-squared score: ', round(reg.score(feature_train,target_train), 3)
print ''
print 'Stats on Test dataset'
print 'r-squared score: ', round(reg.score(feature_test,target_test), 3)




try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="g")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()




### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#apply min max scaling
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

k = 10
def generate_k_best(data_dict, features_list, k):
    #Run SelectKbest - returns a dictionary of best features
    select_k_best = SelectKBest(k=k)
    select_k_best.fit(features, labels)
    scores = select_k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features

k_best = generate_k_best(data_dict, features_list, k)
print "{0} best features are: {1}\n".format(k, k_best.keys())

limited_features = ['poi']
for key in k_best.keys():
    limited_features.append(key)

features_list = limited_features

pretty.pprint (features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(n_splits = 1000, test_size=0.3, random_state=42)   
 
for train_index, test_index in sss.split(X, y):
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test = y[train_index], y[test_index]



#DecisionTreeClassifier Tuning
parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
svr = DecisionTreeClassifier()
clf = GridSearchCV(svr, parameters, scoring = 'f1')
clf.fit(features_train, labels_train)
cri1 = clf.best_params_['min_samples_split']
print cri1

clf = DecisionTreeClassifier(min_samples_split=cri1)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print 'Decision Tree Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)
print '\nFeature Ranking: '
for i in range(k):
    print "{} feature {} ({})".format(i+1,features_list[i+1],round(importances[indices[i]],4))



clf = GaussianNB()
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print 'Gaussian Naive Bayes Classifier'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



#SVC Tuning
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters, scoring = 'f1')
clf.fit(features_train, labels_train)
cri1 = clf.best_params_['kernel']
cri2 = clf.best_params_['C']
print cri1, cri2

clf = SVC(kernel = cri1, C = cri2)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'Support Vector Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



#KNeighbors Tuning
parameters = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
    'leaf_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'metric': ['euclidean', 'minkowski']}
svr = KNeighborsClassifier()
clf = GridSearchCV(svr, parameters, scoring = 'f1')
clf.fit(features_train, labels_train)
cri1 = clf.best_params_['n_neighbors']
cri2 = clf.best_params_['algorithm']
cri3 = clf.best_params_['leaf_size']
cri4 = clf.best_params_['metric']
print cri1, cri2, cri3, cri4



clf = KNeighborsClassifier(n_neighbors = cri1, algorithm = cri2, leaf_size = cri3, metric = cri4)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'K-Neighbors Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



#RandomForest Tuning
parameters = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200], 
    'criterion': ['gini', 'entropy']}
svr = RandomForestClassifier()
clf = GridSearchCV(svr, parameters, scoring = 'f1')
clf.fit(features_train, labels_train)
cri1 = clf.best_params_['n_estimators']
cri2 = clf.best_params_['criterion']
print cri1, cri2

clf = RandomForestClassifier(n_estimators = cri1, criterion = cri2)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'Random Forest Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



#AdaBoost Tuning
parameters = {
    'n_estimators': [5, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000],
    'random_state': [42, 101, 202, 303],
    'learning_rate': [1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}
svr = AdaBoostClassifier()
clf = GridSearchCV(svr, parameters, scoring = 'f1')
clf.fit(features_train, labels_train)
cri1 = clf.best_params_['n_estimators']
cri2 = clf.best_params_['random_state']
cri3 = clf.best_params_['learning_rate']
cri4 = clf.best_params_['algorithm']
print cri1, cri2, cri3, cri4


clf = AdaBoostClassifier(n_estimators = cri1, random_state = cri2, learning_rate = cri3, algorithm = cri4)
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'AdaBoost Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



#Re-running classifier to use for final tesing script.

clf = RandomForestClassifier(n_estimators = 80, criterion = 'gini')
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)

print 'Random Forest Classifier - Tuned'
print 'Accuracy:', round(accuracy_score(prediction, labels_test),4)
print 'Precision:', round(precision_score(prediction, labels_test),4)
print 'Recall:', round(recall_score(prediction, labels_test),4)
print 'F1 Score:', round(f1_score(prediction, labels_test),4)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)