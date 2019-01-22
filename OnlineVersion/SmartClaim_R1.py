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
import glob
import pickle
import datetime
import csv

#%matplotlib inline 
import matplotlib.pylab as plt

#from bokeh.charts import Bar, BoxPlot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import confusion_matrix

#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier

# import model modules
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
#from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# <codecell>

# load functions from python file
from blagging import BlaggingClassifier
from functions import _pull_hanaTable
from functions import _create_path
from functions import _create_filename
from functions import _class_info
from functions import _verify_claim
from functions import _plot_ROC_curve
from functions import _replace_inf
from functions import _plot_prob
from functions import _plot_decision_tree
from functions import _plot_confusion_matrix
from functions import _predict_claim
from functions import _encode_label_
from functions import _convert_float

# load models
from models import label_encoder
    
# load constants
from constants import INPUT_TAGS

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
# set working directory
parentDir = 'C:/Users/E154709/SmartClaim'
os.chdir(parentDir)

    # extract the basename
fprogram = 'SmartClaim.py'
curFname,file_extension1 = os.path.splitext(fprogram)
print(curFname)

#modeling data
dataPath = parentDir + '/Data/'
_create_path(dataPath)

#modeling data
inputPath = parentDir + '/Input/'
_create_path(inputPath)

allFiles = glob.glob(inputPath + "/*.csv")
 
# <codecell>   
# main program to load data, build model, validation, testing...
def main(finput, imode = 1):
        
    """
    # finput: input data file
    # outputPath: output path
    # imode = 0: training, 1: prediction
    """
    
    fbasename = os.path.basename(finput)
    fname, file_extension2 = os.path.splitext(fbasename)
    parentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory    
    print(parentDir)
    
    #output directory
    outputPath = parentDir + '/Output/'
    _create_path(outputPath)
    
    #output directory
    modelPath = parentDir + '/Model/'
    _create_path(modelPath)
    
    #output directory
    statePath = parentDir + '/State/'
    _create_path(statePath)
    
    # save encoder  
    affix = ".pkl"
    prename = 'encoder_'
    fencoder = _create_filename(modelPath, curFname, prename, affix)
    
    # save normalization scaler    
    affix = ".pkl"
    prename = 'normScaler_'
    fnormScaler = _create_filename(modelPath, curFname, prename, affix)
    
    # save model   
    affix = ".pkl"
    prename = 'clf_'
    fmodel = _create_filename(modelPath, curFname, prename, affix)
    
     # save model   
    affix = ".log"
    prename = 'run_'
    fstate = _create_filename(statePath, curFname, prename, affix)
         
            
    # <codecell>
    if imode == 0:                 
        
        print('\n imode=0: Model training... \n')
        # select the model used for prediction in QA
        
        # <codecell>
        # read input data
        modelData = pd.read_csv(finput, error_bad_lines=False, thousands=',') #encoding='latin1') # Read the data 
        # import input tags
        modelData_inf = _replace_inf(modelData)
        X = modelData_inf[INPUT_TAGS]
        
        # read model data
        yTags = 'Claims'
        y = modelData_inf[yTags]
        
        # <codecell>
        """
        Preprocessing data
        
        """
        # split data into modeling and validation sets
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
        
        #save scaler into file    
        with open(fnormScaler, "wb") as fo:
            pickle.dump(normX_scaler, fo)
        
        
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
        """
        Modeling using scaled data
        """
        
        # predince model parameters
        xm = nxModel.iloc[:, 3:5]
        #xm = pd.DataFrame(xm)
        print(xm.head(2))
        #n_features = X.shape[1]
        C = 1
        #kernel = 1.0 * RBF(1.0)
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
        # classifiers comparison using full data sets
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
        # be aware that it is very slow to create the learning curves
        """
        title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        
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
        
        #save scaler into file    
        with open(fmodel, "wb") as fo:
            pickle.dump(aclf, fo)
                
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
        
        # check which classifier is available for predict_proba...
        from sklearn.utils.testing import all_estimators
        
        estimators = all_estimators()
        
        for name, class_ in estimators:
            if hasattr(class_, 'predict_proba'):
                print(name)
    # <codecell>
    else: # prediction
        
        print('\n imode=1: Model predicting... \n')
        
        print('\n ... Extract data table from hana server')
        modelData = _pull_hanaTable()
                
        if modelData.empty:
            with open(fstate, "w") as text_file:
                print("\n****************************", file=text_file)
                print("*...No any new claim now...*", file=text_file)
                print("****************************\n", file=text_file)    
            return
        
        else:
            with open(fstate, "w") as text_file:
                print("\n ... Extract data table from hana server", file=text_file)
                       
            # save data   
            affix = ".csv"
            prename = 'data_'
            now = datetime.datetime.now()
            fdata = _create_filename(dataPath, now.strftime("%Y-%m-%d"), prename, affix)
            
            with open (fdata, 'w') as csvfile:
                #csvwriter = csv.writer(csvfile, delimiter=',')
                #csvwriter.writenow(modelData)
                modelData.to_csv(csvfile)
                
                        
        print('\n ... Data preprocessing ...')
        
        modelData.rename(columns={'TOTAL_CLAIM_CNT': 'numClaims', 'TRX_CNT': 'numSales', 
                                  'TOTAL_CLAIM':'dollarClaims', 'NET_SALES':'dollarSales'}, inplace=True)
        tagList = ['numClaims', 'numSales','dollarClaims', 'dollarSales']
        
        for item in tagList:
            modelData = _convert_float(modelData, item)
        
        modelData['dollarPerClaim'] = modelData['dollarClaims']/modelData['numClaims']
        modelData['dollarPerSale'] = modelData['dollarSales']/modelData['numSales']
        modelData['claimRatio'] = modelData['dollarClaims']/modelData['dollarSales']
                
        # import input tags
        modelData_inf = _replace_inf(modelData)
        X = modelData_inf[INPUT_TAGS]
        
        # <codecell>
        print('\n ... Product property encodeing...')
                
        # load encoder parameters created from modeling data
        with open(fencoder, 'rb') as fo:
            encoder_style, encoder_back, encoder_size, encoder_color = pickle.load(fo)
            
        #encodeing
        tag_style = 'INVENTORY_STYLE_CD'
        X, numb_style = _encode_label_(X, tag_style, encoder_style)
        
        tag_back = 'INVENTORY_BACKING_CD'
        X, numb_back = _encode_label_(X, tag_back,encoder_back)
        
        tag_size = 'INVENTORY_SIZE_CD'
        X, numb_size = _encode_label_(X, tag_size, encoder_size)
        
        tag_color = 'INVENTORY_COLOR_CD'
        X, numb_color = _encode_label_(X, tag_color, encoder_color)
        
        # <codecell>
        
        print('\n ... Input data normalization ...')
        
        # load normalization scaler created from modeling data
        with open(fnormScaler, 'rb') as fo:
            XnormScaler = pickle.load(fo)
            
        xnTst = XnormScaler.transform(X.values)  # normalize
        xnTst = pd.DataFrame(xnTst)
        xnTst.index = X.index
        xnTst.columns = X.columns
        
            
        
        # <codecell>
        
        print('\n ... Model prediction ...')
        
        # load trained clf model
        with open(fmodel,'rb') as fo:
            clf = pickle.load(fo)

        pred_output = _predict_claim(clf, xnTst)
        
        pred_op = pd.concat([modelData, pred_output], axis=1)
        
        # save data   
        affix = ".csv"
        prename = 'output_'
        #now = datetime.datetime.now()
        foutput = _create_filename(outputPath, now.strftime("%Y-%m-%d"), prename, affix)        
        with open (foutput, 'w') as csvfile:
            pred_op.to_csv(csvfile)
        
        savePath = "//vchgtbwp101a.extdmz.ad/Tableau$/Flooring/RVP Reporting/Claim_Scoring"
        fsave = _create_filename(savePath, curFname, prename, affix)
        with open (fsave, 'w') as csvfile:
            pred_op.to_csv(csvfile)

# <codecell>
if __name__ == '__main__':
    
    imode = 1
    for file_ in allFiles:    
        main(file_, imode)    