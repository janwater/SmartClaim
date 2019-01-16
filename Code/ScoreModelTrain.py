# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:35:04 2017

@author: E154709
"""

"""
Carpet Claim Intelligent Recomendation Modeling
 - Based on cusomter historical data (Sales, Claims, $Sales and $Claims) - R1
 - Product information (Style, Size, Backing, & Color)
 - Machine learning
 - Automatic and quantitative recomendation of the claim for approval or declination
 
"""
print(__doc__)

#
# Load necessary libraries
import numpy as np
import pandas as pd
import os, inspect
import csv
#%matplotlib inline 
import pylab
import matplotlib.pylab as plt
import matplotlib.image as img
import scipy.stats as stats
import seaborn as sns
from IPython.core.display import HTML 
from sklearn.preprocessing import scale
from bokeh.io import show, output_file
from bokeh.plotting import figure
#from bokeh.charts import Bar, BoxPlot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import Counter
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import column_or_1d
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# <codecell>
# load function from python file
from blagging import BlaggingClassifier
from csModel import _create_filename
from csModel import _read_inputTags
from csModel import _class_info
from csModel import _verify_claim
from csModel import _plot_ROC_curve
from csModel import _replace_inf
from csModel import _plot_prob
from csModel import _plot_decision_tree
from csModel import _plot_confusion_matrix
from csModel import _plot_learning_curve
from csModel import plotPCA
   # <codecell>     
def _plot_PCA(data, y, tag):
    #minmax to normalize X
    normX = plotPCA.minmax_X(data)
    dpca= plotPCA.do_PCA(normX)
    plotPCA.plot_PCA(dpca, y, tag)
    plotPCA.pca_summary(dpca, out=True)
    plotPCA.plot_pca_var(dpca)
    return normX, dpca

# <codecell>
# set working directory
parentDir = 'C:/Users/E154709/Desktop/Mohawk/Claims'
os.chdir(parentDir)

#output directory
outputPath = parentDir + '/Output/'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# code directory
codePath = parentDir + '/Code/'
if not os.path.exists(codePath):
    os.mkdir(codePath)

#modeling data
dataPath = parentDir + '/Data/'
if not os.path.exists(dataPath):
    os.mkdir(dataPath)
    
#create data file
fname = 'modelData2'
fmodelData = _create_filename(dataPath,fname)
print(fmodelData)
# <codecell>   
"""
# Load modeling data

"""
modelData = pd.read_csv(fmodelData, encoding='latin1') # Read the data
print(modelData.columns, modelData.head(5))

modelData.isnull().any()
# <codecell>
# import input tags
inputTags = _read_inputTags()
yTags = 'Claims'
modelData_inf = _replace_inf(modelData)
# <codecell>

# read model data
X = modelData_inf[inputTags]
y = modelData_inf[yTags]

# <codecell>
print(modelData_inf.head(5))

# <codecell>
"""
Preprocessing data

"""
# split data
ts = 0.3
xTrain, xTest, yTrain, yTest = train_test_split(X, y, stratify=y, test_size=ts, random_state=531)  

# <codecell>
# normalize data
xvalue = xTrain.values 
normX_scaler = preprocessing.StandardScaler().fit(xvalue) 
#normX_scaler = preprocessing.MinMaxScaler().fit(xvalue) 

xScaled = normX_scaler.transform(xvalue)   
xScaled_test = normX_scaler.transform(xTest.values) 
xnTrain = pd.DataFrame(xScaled)
xnTest = pd.DataFrame(xScaled_test)

xnTrain.index = xTrain.index
xnTrain.columns = xTrain.columns
xnTest.index = xTest.index
xnTest.columns = xTest.columns

# <codecell>
# class information
_class_info(yTrain)
_class_info(yTest)

# <codecell>
#split the training data into modeling set and validation set
nxModel, nxValid, yModel, yValid = train_test_split(xnTrain, yTrain, stratify=yTrain, test_size=ts, random_state=531)

# <codecell>
_class_info(yModel)
_class_info(yValid)

# <codecell>
# PCA of data
criticTags = np.array(inputTags)
yrTrain = column_or_1d(yModel)
critXtrn = _plot_PCA(nxModel, yrTrain, 'Critic Tags')

# <codecell>
# set number of principal components for analysis
from sklearn.decomposition import PCA
ncomp = 5
pca = PCA(n_components = ncomp)
pca.fit(xnTrain)
dpcaTrn1 = pca.transform(xnTrain)

colnames = ["PC"+str(i) for i in range(1, ncomp+1)]
dpcaTrain = pd.DataFrame(dpcaTrn1, index = xnTrain.index, columns = colnames)

# <codecell>
#split the training data into modeling set and validation set
dpcaModel, dpcaValid, yModel, yValid = train_test_split(dpcaTrain, yTrain, stratify=yTrain, test_size=ts, random_state=531)

# using testing data PCA for testing data
#pca.fit(xTest)
dpcaTst1 = pca.transform(xnTest)
dpcaTest = pd.DataFrame(dpcaTst1, index = xnTest.index, columns = colnames)
predictors = colnames
# <codecell>
"""
Modeling using scaled data
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
#from sklearn.utils.validation import column_or_1d
#from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
#from elm import ELMClassifier, ELMRegressor, GenELMClassifier, GenELMRegressor
#from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
# <codecell>
# predince model parameters
xm = nxModel.iloc[:, 3:5]
#xm = pd.DataFrame(xm)
print(xm.head(2))
n_features = X.shape[1]
C = 1
kernel = 1.0 * RBF(1.0)
prior = [0.3, 0.7] # set the priors for classification

# <codecell>
# Create different classifiers. The logistic regression cannot do
# multiclass out of the box.
classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
               'RBF SVC': SVC(kernel='rbf', C=10, gamma=2, class_weight='balanced',probability=True,
                              random_state=0),
               'Linear SVC': SVC(kernel='linear', C=C, class_weight='balanced',probability=True,
                                 random_state=0),
               'L2 logistic (Multinomial)': LogisticRegression(
                       C=C, solver='lbfgs', multi_class='multinomial'),
               'Naive Bayes': GaussianNB(priors = prior),
               #'Blag': BlaggingClassifier(),
               'Nearest Neighbors': KNeighborsClassifier(4),
               'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight = 'balanced'),
               'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=10, 
                                                       max_features=1, class_weight = 'balanced'),
               'Nueral Network': MLPClassifier(alpha=1, activation = 'logistic', learning_rate = 'adaptive'),
               'AdaBoost': AdaBoostClassifier(),
               'QDA': QuadraticDiscriminantAnalysis(),
               'Bag': BaggingClassifier(),
               'Blag': BlaggingClassifier()
               #'GPC': GaussianProcessClassifier(kernel)  too time-consuming
               }

n_classifiers = len(classifiers)

# <codecell>
f = plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=.2, top=.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(xm, yModel)

    y_pred = classifier.predict(xm)
    classif_rate = np.mean(y_pred.ravel() == yModel.ravel()) * 100
    print("classif_rate for %s : %f " % (name, classif_rate))

    # View probabilities=
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(xm.iloc[idx,0], xm.iloc[idx,1], marker='o', c='k')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')
f.tight_layout()
plt.show()

# <codecell>
# classifiers comparison
for index, (name, classifier) in enumerate(classifiers.items()):
    print('\n Modeling...')
    classifier.fit(nxModel, yModel)
    print('\n Validation...')
    res_pred = _verify_claim(classifier, name, nxValid, yValid)
    print('\n Testing...')
    res_test = _verify_claim(classifier, name, xnTest, yTest)

# plot probability curves
_plot_prob(classifiers, nxModel, yModel, nxValid, yValid)

# <codecell>

#defube a function to output validation and testing results
def _output_val_test(clf, name, nxValid, yValid, xnTest, yTest):
    
    # validation data
    print("\n Validation...")
    pred = _verify_claim(clf, name, nxValid, yValid)
    print(pred.head(20), pred.shape)
    pred['Index'] = 'val'
    val = pd.concat([nxValid, pred], axis = 1)
    
    # Testing data
    print("\n Testing...")
    test = _verify_claim(clf, name, xnTest, yTest)
    print(test.head(20), test.shape)
    test['Index'] = 'tst'
    tst = pd.concat([xnTest, test], axis = 1)
    
    output = pd.concat([val, tst], axis = 0)
    print(output.shape, output.head(5))
    
    return output
# <codecell>
print(res_pred.head(20), res_pred.shape)
test_results = ['Declined', 'Paid'] 
# <codecell>
# C-Support Vector Classifier...
print("\nC-Support Vector Classifier...")

# modeling
clf_svc = _plot_ROC_curve(SVC(C= 10, kernel = 'rbf', class_weight='balanced', probability = 1), nxModel, yModel)
name = 'RBF SVC'

svc_output = _output_val_test(clf_svc, name, nxValid, yValid, xnTest, yTest)
svc_output.to_csv('svc_output.csv')

# <codecell>
_class_info(svc_output.Claims)
_class_info(svc_output.Pred)

# <codecell>
#Load up a BlaggingClassifier (balanced bagging) implementing the technique of Wallace et al.

# Blagging Classifier...
#print("\n Blagging Classifier...")
# modeling

print("\n Blagging Modeling...")
clf_blag = _plot_ROC_curve(BlaggingClassifier(), nxModel, yModel)
name = 'Blagging'

# validation data
print("\n Blagging Validation...")
blag_pred = _verify_claim(clf_blag, name, nxValid, yValid)
print(blag_pred.head(20), blag_pred.shape)

# Testing data
print("\n Blagging Testing...")
blag_test = _verify_claim(clf_blag, name, xnTest, yTest)
print(blag_test.head(20), blag_test.shape)

# <codecell>

# Random Forest Classifier...
#print("\n Random Forest Classifier...")
# modeling
print("\n Random Forest Modeling...")
clf_rf = _plot_ROC_curve(RandomForestClassifier(max_depth=10, n_estimators=100,
                                                max_features=1, class_weight = 'balanced'), nxModel, yModel)
name = 'RF'

rf_output = _output_val_test(clf_rf, name, nxValid, yValid, xnTest, yTest)

rf_output.to_csv('rf_output.csv')

# <codecell>
print("\n X Gradient Boosting Modeling...")

org_params = {'n_estimators': 500,
              'max_leaf_nodes': 2, 
              'max_depth': 5,
              'min_samples_split': 2,
              'learning_rate': 0.01
              }
params = dict(org_params)
#clf_gbrt, msError = GBRT_pred(org_params, nxModel, yModel, nxValid, yValid)
clf_gbrt = _plot_ROC_curve(ensemble.GradientBoostingClassifier(**params), nxModel, yModel)
name = 'GBRT'

gbrt_output = _output_val_test(clf_gbrt, name, nxValid, yValid, xnTest, yTest)
gbrt_output.to_csv('gbrt_output.csv')
# <codecell>

# modeling
print("\n Decision Tree Modeling...")
clf_dt = _plot_ROC_curve(DecisionTreeClassifier(max_depth=5, class_weight = 'balanced'), nxModel, yModel)
name = 'DT'

dt_output = _output_val_test(clf_dt, name, nxValid, yValid, xnTest, yTest)
dt_output.to_csv('dt_output.csv')
# <codecell>

# modeling
print("\n Extra Decision Tree Modeling...")
clf_edt = _plot_ROC_curve(ExtraTreesClassifier(n_estimators = 16, max_depth=3, class_weight = 'balanced'), nxModel, yModel)
name = 'EDT'

edt_output = _output_val_test(clf_edt, name, nxValid, yValid, xnTest, yTest)
edt_output.to_csv('edt_output.csv')
# <codecell>

# overfitting on paid
# modeling
print("\n AdaBoost Modeling...")
clf_ab = _plot_ROC_curve(AdaBoostClassifier(n_estimators=200), nxModel, yModel)
name = 'AB'

ab_output = _output_val_test(clf_ab, name, nxValid, yValid, xnTest, yTest)
ab_output.to_csv('ab_output.csv')
# <codecell>

# modeling
print("\n Nearest Neighbor Modeling...")
clf_knn = _plot_ROC_curve(KNeighborsClassifier(4), nxModel, yModel)
name = 'kNN'

knn_output = _output_val_test(clf_knn, name, nxValid, yValid, xnTest, yTest)
knn_output.to_csv('knn_output.csv')

# <codecell>

# modeling
print("\n Naive Bayesian Modeling...")
clf_gnb = _plot_ROC_curve(GaussianNB(priors = prior), nxModel, yModel)
name = 'GNB'

gnb_output = _output_val_test(clf_gnb, name, nxValid, yValid, xnTest, yTest)
gnb_output.to_csv('gnb_output.csv')

# <codecell>

# modeling - similar to SVM, but time-consumsing
"""
print("\nNu Support Vector Machine Modeling...")

clf_nsvc = _plot_ROC_curve(NuSVC(kernel = 'rbf', class_weight='balanced', probability = 1), nxModel, yModel)
name = 'NuSVC'

nsvc_output = _output_val_test(clf_nsvc, name, nxValid, yValid, xnTest, yTest)
nsvc_output.to_csv('nsvc_output.csv')
"""
print("\n SGD Classifier Modeling...")
#Using loss="log" or loss="modified_huber" enables the predict_proba method, 
#which gives a vector of probability estimates 
clf_sgd = _plot_ROC_curve(SGDClassifier(loss= 'modified_huber', 
                                        #class_weight='balanced',
                                        penalty="l1"), nxModel, yModel)
name = 'SGD'

sgd_output = _output_val_test(clf_sgd, name, nxValid, yValid, xnTest, yTest)
sgd_output.to_csv('sgd_output.csv')
# <codecell>
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
"""
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
_plot_learning_curve(estimator, title, nxModel, yModel, ylim=(0.6, 0.75), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.01, C= 10, kernel = 'rbf', class_weight='balanced', probability = 1)
_plot_learning_curve(estimator, title, nxModel, yModel, (0.5, 0.7), cv=cv, n_jobs=4)

plt.show()
"""
# <codecell>

# modeling
print("\n Voting Classifier Modeling...")
eclf = VotingClassifier(estimators=[('edt', clf_edt), ('blag', clf_blag), ('svc', clf_svc)], voting='soft', weights=[1,2,2])
eclf = eclf.fit(nxModel, yModel)
name = 'Voting'

eclf_output = _output_val_test(eclf, name, nxValid, yValid, xnTest, yTest)
eclf_output.to_csv('eclf_output.csv')
# <codecell>

# modeling
print("\n Voting Classifier Modeling...")
aclf = VotingClassifier(estimators=[('dt', clf_dt), ('blag', clf_blag), ('svc', clf_svc)], voting='soft', weights=[1,3,2])
aclf = aclf.fit(nxModel, yModel)
name = 'Voting'

aclf_output = _output_val_test(aclf, name, nxValid, yValid, xnTest, yTest)
aclf_output.to_csv('aclf_output.csv')

# <codecell>
# Plotting decision tree
feature_names=nxModel.columns.values
class_names=test_results

graph = _plot_decision_tree(clf_dt,feature_names, class_names)  
graph 

# <codecell>
y_test = aclf_output['Claims']
y_pred = aclf_output['Pred']
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
_plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
_plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# <codecell>

"""
us scaled PCA data
"""

# <codecell>
# classifiers comparison
for index, (name, classifier) in enumerate(classifiers.items()):
    print('\n Modeling...')
    classifier.fit(dpcaModel, yModel)
    print('\n Validation...')
    res_pred = _verify_claim(classifier, name, dpcaValid, yValid)
    print('\n Testing...')
    res_test = _verify_claim(classifier, name, dpcaTest, yTest)

# plot probability curves
_plot_prob(classifiers, dpcaModel, yModel, dpcaValid, yValid)

# <codecell>
#Load up a BlaggingClassifier (balanced bagging) implementing the technique of Wallace et al.

# Blagging Classifier...
#print("\n Blagging Classifier...")
# modeling

print("\n Blagging Modeling...")
clf_blag = _plot_ROC_curve(BlaggingClassifier(), dpcaModel, yModel)
name = 'Blagging'

blag_output = _output_val_test(clf_blag, name, dpcaValid, yValid, dpcaTest, yTest)
blag_output.to_csv('blag_output.csv')
# <codecell>
# Bagging Classifier...
print("\n Bagging Classifier...")

# modeling
clf_bag = _plot_ROC_curve(BaggingClassifier(), dpcaModel, yModel)
name = 'BAG'

bag_output = _output_val_test(clf_bag, name, dpcaValid, yValid, dpcaTest, yTest)
bag_output.to_csv('bag_output.csv')
# <codecell>
# C-Support Vector Classifier...
print("\nC-Support Vector Classifier...")

# modeling
clf_svc = _plot_ROC_curve(SVC(C= 10, kernel = 'rbf', class_weight='balanced', probability = 1), dpcaModel, yModel)
name = 'RBF SVC'

svc_output = _output_val_test(clf_svc, name, dpcaValid, yValid, dpcaTest, yTest)
svc_output.to_csv('svc_output.csv')
# <codecell>

# Random Forest Classifier...
#print("\n Random Forest Classifier...")
# modeling
print("\n Random Forest Modeling...")
clf_rf = _plot_ROC_curve(RandomForestClassifier(max_depth=10, n_estimators=100,
                                                max_features=1, class_weight = 'balanced'), dpcaModel, yModel)
name = 'RF'

rf_output = _output_val_test(clf_rf, name, dpcaValid, yValid, dpcaTest, yTest)

rf_output.to_csv('rf_output.csv')
# <codecell>

# modeling
print("\n Voting Classifier Modeling...")
rclf = VotingClassifier(estimators=[('rf', clf_rf), ('blag', clf_blag), ('svc', clf_svc)], voting='soft', weights=[1,2,2])
rclf = rclf.fit(dpcaModel, yModel)
name = 'Voting'

rclf_output = _output_val_test(rclf, name, dpcaValid, yValid, dpcaTest, yTest)
rclf_output.to_csv('rclf_output.csv')
# <codecell>

# check which classifier is available for predict_proba...
from sklearn.utils.testing import all_estimators

estimators = all_estimators()

for name, class_ in estimators:
    if hasattr(class_, 'predict_proba'):
        print(name)

# <codecell>