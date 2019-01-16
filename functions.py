# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:52:43 2017

@author: E154709
"""

"""
AI claim scoring model files
"""

# Load modules
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.utils.validation import column_or_1d
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from scipy import interp
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV,ShuffleSplit, cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn import tree
import graphviz 
import itertools
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from matplotlib.pyplot import subplots, show
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pyodbc
import pyhdb
from string import whitespace

# load models
from models import label_encoder

# load constants
from constants import DROPLIST, TAGLIST_CON
# tags created for input variables
from constants import COMTAGS, MUCOMTAGS, TAGZONETEMP

# <codecell>
# functions

#used in server: vchgpcpp101a
def _pull_hana_odbc():
    #pull data from Hana table
     
    conn = None
    #cursor = None
    conn = pyodbc.connect (
                            DSN = 'HANA PROD',
                            uid = 'MART_RDR_2',
                            pwd = 'Re@d3r13')
    
    df = pd.read_sql_query("""select * from MHK_WORKSPACE.FACT_CLAIMS_AI""",conn)
    conn.close()
    
    return df

# used in laptop
def _pull_hana_hdb():
    #pull data from Hana table
     
    conn = None
    #cursor = None 
    
    conn = pyhdb.connect(
            host="cchgpasdb2",
            port=32553,
            user="MART_RDR_2",
            password="Re@d3r13"
            )
    
    df = pd.read_sql_query("""select * from MHK_WORKSPACE.FACT_CLAIMS_AI""",conn)
    conn.close()
    
    return df

# process attribute tags
def _check_mantags(df, MANTAGS):
           
    floatTags = ['FZTSPI', 'FZSWPT']
    for tag in MANTAGS:
        if tag in floatTags:
            df[tag] = df[tag].astype(float)
        else:
             # force to be string
             df[tag] = df[tag].astype(str)  
             # remove space in the tag string
             df = remove_space(df, tag)
    
    return df


def _class_info(classes):
    # count class information
    counts = Counter(classes)
    total = sum(list(counts.values()))
  
    for cls in counts.keys():
        print("%6s: % 7d  =  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))


def clean_df(df, tagSel):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    #df.dropna(inplace=True)
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    df.fillna(0,inplace=True) 
    
    # output error code
    #df['FZTSPI'] = df['FZTSPI'].astype(float)
    #df['FZSPWT'] = df['FZSPWT'].astype(float)
    errCode = []
    if tagSel == 0: 
        errCode1 = []
        errCode2 = []
        errCode1 = [1 if x is None else 0 for x in df['FZTSPI']]      
        errCode2 = [1 if 'nan' in x else 0 for x in df['FZSPWT']]
              
        errCode1 = pd.DataFrame(errCode1, index=df.index, columns = ['ErrorCode'])
        errCode2 = pd.DataFrame(errCode2, index=df.index, columns = ['ErrorCode'])
            
        if errCode1 is None:
            errCode1['ErrorCode'] = 0
        if errCode2 is None:
            errCode2['ErrorCode'] = 0    
        
        errCode = errCode1 + errCode2
        errCode = pd.DataFrame(errCode, index=df.index, columns = ['ErrorCode'])
    
    #fill nan
    df = df.replace('nan', np.nan)
    df.fillna(0,inplace=True) 
        
    return df, errCode #df[indices_to_keep].astype(np.float64)


def clean_tags(tags):
    # clean sensor Tags
    sensorTags = _remove_Tags(tags)
    print(len(sensorTags))
    
    # create xTags
    xTags, manTags = _create_sensorTags()
    for tag in xTags:
        if tag not in sensorTags:
            xTags.remove(tag)
    print(len(xTags))
    
    for tag in manTags:
        if tag not in sensorTags:
            manTags.remove(tag)
    print(len(manTags))
    
    return xTags, manTags


def _convert_float(data, tag):
    df = data[tag]
    float_list = [float(x) for x in df]
    #float_list = pd.DataFrame(float_list, index = data.index, columns = data.columns)
    data[tag]=float_list
    return data


def _create_filename(path, fname, prefix = '', affix = '.csv'):
    # create new file name
    output_name = prefix + fname + affix        
    foutput_name = os.path.join(path, '', output_name)
             
    return foutput_name


def _create_path(path):
    if not os.path.exists(path):
        os.mkdir(path) 


def _create_sensorTags():
    """
    Zone temperature tags processing
     - 15 temp tags in zone (zone 14 excluded)
     - 18 features for each tags
     - PCA preprocessing to address correlation for all 18 features
    
    """
    sensorTags = []
    manTags = []
    featureList =['mu','std','max','min','med','var','skew','kurt','cwtp','nrng', 
                  'mabch','much','sabsch', 'bmu', 'amu', 'ent', 'ac', 'q75']
    
    # for specific mu tags
    for item in featureList[0:5]:
        comTags = [item + '_' + myitem for myitem in MUCOMTAGS]
        for tag in comTags:
            sensorTags.append(tag)
    
    # for generic tags
    for item in featureList[5:]:
        comTags = [item + '_' + myitem for myitem in COMTAGS]
        for tag in comTags:
            sensorTags.append(tag)
    
    #Total 18 features for each variable (15 temperature tags excluding no 14)        
    for item in featureList:        
        tagZoneTemps = [item + '_' + myitem for myitem in TAGZONETEMP]
        for tag in tagZoneTemps:
            sensorTags.append(tag)
    
        # for generic tags
    for item in featureList[4:9]:
        mTags = [item + '_' + myitem for myitem in TAGLIST_CON]
        for tag in mTags:
            manTags.append(tag)
            
    sensorTags = list(sensorTags)
    manTags = list(manTags)
    return sensorTags, manTags
        


def _encode_data(df):

    tag = 'F1BACK'
    df_uniquen, encoder_back_ = _encode_label_(df, tag)
       
    tag = 'F1STYL'
    df_uniquen, encoder_style_ = _encode_label_(df_uniquen, tag)
    
    tag = 'FZGAGE'
    df_uniquen, encoder_gauge_ = _encode_label_(df_uniquen, tag)
    
    tag = 'F1CLR'
    df_uniquen, encoder_color_ = _encode_label_(df_uniquen, tag)
    
    encoder_ = [encoder_back_, encoder_style_, encoder_gauge_, encoder_color_]
    
    return df_uniquen, encoder_, 

#use sklearn.preprocessing.LabelEncoder
def _encode_label(data, tag, encoder = None):
    if encoder is None: 
        encoder = LabelEncoder()
        encoder.fit(data[tag].astype('str'))
    
    encode_label = encoder.transform(data[tag].astype('str'))    
    
    data.loc[:,tag] = encode_label
    return data, encoder


# use self-defined label_encoder
def _encode_label_(data, tag, encoder = None):
    df = pd.DataFrame(data[tag], index = data.index, columns=[tag])
    
    if encoder is None:
        encoder = label_encoder()
        encoder.fit_pd(df)

    encode_label = encoder.transform_pd(df)
    
    data.loc[:,tag] = encode_label
    return data, encoder


#define the GBRT regression model
def GBRT_pred(org_params, xTrain, yTrain, xTest, yTest):
    #generate the model
    params = dict(org_params)
    model = ensemble.GradientBoostingClassifier(**params)
    model.fit(xTrain, yTrain)
    
    # compute mse on test set
    msError = []
    predictions = model.staged_predict(xTest)
    
    for p in predictions:
        msError.append(mean_squared_error(yTest, p))

    return model, msError


#Defining this as a function so we can call it anytime we want. 
def GradientBooster(param_grid, xTrain, yTrain, n_jobs = 5): 
    estimator = ensemble.GradientBoostingClassifier() 

    #Choose cross-validation generator - let's choose ShuffleSplit which randomly 
    #shuffles and selects Train and CV sets for each iteration. There are other 
    #methods like the KFold split. 
    cv = ShuffleSplit(n_splits=10, test_size=0.2,random_state=0) 

    #Apply the cross-validation iterator on the Training set using GridSearchCV. 
    #This will run the classifier on the different train/cv splits using parameters 
    #specified and return the model that has the best results 

    #Note that we are tuning based on the F1 score 2PR/P+R where P is Precision and 
    #R is Recall. This may not always be the best score to tune our model on. 
        
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs) 

    #Also note that we're feeding multiple neighbors to the GridSearch to try out. 
    #We'll now fit the training dataset to this classifier 
    classifier.fit(xTrain, yTrain) 

    #Let's look at the best estimator that was found by GridSearchCV 
    model = classifier.best_estimator_
    print ("Best Estimator learned through GridSearch")
    print (model) 
    return model 


def _normalize_X(X, normX_scaler = None, scalerInd = 1, tagSel=0):
    """
    Normalize the given data frame to a standardized zero mean and deviation
    
    X = data frame or arrary
    normScaler = 1: StandardScaler, otherwise, minMaxScaler

    """
    X, errCode = clean_df(X, tagSel)
    
    xvalue = X.values
    
    if normX_scaler is None:
        if scalerInd ==1:
            normX_scaler = preprocessing.StandardScaler().fit(xvalue) 
        else:
            normX_scaler = preprocessing.MinMaxScaler().fit(xvalue) 
        
    xScaled = normX_scaler.transform(xvalue)
    normX = pd.DataFrame(xScaled)
    
    normX.index = X.index
    normX.columns = X.columns
    return normX, normX_scaler, errCode 

#from contextlib import contextmanager
#@contextmanager
def _open_file(path, mode):
    the_file = open(path, mode)
    yield the_file
    the_file.close()



#defube a function to output validation and testing results
def _output_val_test(clf, name, nxValid, yValid, xnTest, yTest, test_results):
    
    # validation data
    print("\n Validation...")
    pred, recall_val = _verify_claim(clf, name, nxValid, yValid, test_results)
    #print(pred.head(20), pred.shape)
    pred['Index'] = 'val'
    val = pd.concat([nxValid, pred], axis = 1)
    
    # Testing data
    print("\n Testing...")
    test, recall_test = _verify_claim(clf, name, xnTest, yTest, test_results)
    #print(test.head(20), test.shape)
    test['Index'] = 'tst'
    tst = pd.concat([xnTest, test], axis = 1)
    
    output = pd.concat([val, tst], axis = 0)
    #print(output.shape, output.head(5))
    recall = pd.concat([pd.Series(recall_val), pd.Series(recall_test)], axis =0)
    recall = pd.DataFrame(recall)
    return output, recall

 
def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel= 'Predicted label'


def _plot_decision_tree(clf_dt, feature_names, class_names):
    # plot decision tree
    dot_data = tree.export_graphviz(clf_dt, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("Claim Scoring") 
    dot_data = tree.export_graphviz(clf_dt, out_file=None, 
                                feature_names=feature_names,  
                                class_names=class_names,  
                                filled=True, rounded=True,  
                                special_characters=True)  
    graph = graphviz.Source(dot_data)
    
    return graph


def _plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def _plot_prob(classifiers, X_train, y_train, X_test, y_test):
    # #############################################################################
    # Plot calibration plots
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for index, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)
    
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))
    
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

# <codecell>
def plot_PR_curve(classifier, X, y, n_folds=5):
    """
    Plot a basic precision/recall curve.
    """
    plt.figure()
    for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1],
                                                               pos_label=1)
        plt.plot(recall, precision, lw=1, label='PR fold %d' % (i,))
   #  clf_name = str(type(classifier))
   # clf_name = clf_name[clf_name.rindex('.')+1:]
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return classifier


# let's switch to 2-dimensional curves; specifically, the ROC and precision-recall curves.
# Define ROC function to plot ROC-Recall curve.
def _plot_ROC_curve(classifier, X, y, pos_label=1, n_folds=5):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    aucs = []
       
    all_tpr = []
    for i, (train, test) in enumerate(StratifiedKFold(n_splits=n_folds).split(X,y)):    
        #print(i, train, test)
        # use replace function defined below
        xTrain = _replace_inf(X.iloc[train,:])
 #       print(y[train].isnull().any())
        yTrain = _replace_inf(y[train])     
        yTrain = yTrain.fillna(0) # some values may be infinite or too large
#        xTrain_large = xTrain[(xTrain >= 1.7976931348623157e+5).any(axis=1)]       
        
        probas_ = classifier.fit(xTrain.values, yTrain.values).predict_proba(X.iloc[test,:])
        
        # Compute ROC curve and area under the curve        
        yTest = _replace_inf(y[test])
        #print(yTest.head(2))
        
        #print(yTest.isnull().any())
        yTest = yTest.fillna(0)
        #print(yTest.isnull().any())
        
        #yTest = yTest.astype('int')
        fpr, tpr, thresholds = roc_curve(yTest, probas_[:, 0], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        all_tpr.append(mean_tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random', alpha=0.8)
    
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
         lw=2, alpha=0.8)
    
    std_tpr = np.std(all_tpr, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha=0.2,
                     label = r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return classifier

 
def _predict_claim(classifier, X):
    # verify classifier model 
    #y_pred = classifier.predict(xValid)
    
    y_pred = (classifier.predict(X) > 0.5).astype('int')
    
    # Output probabilities
    probas = classifier.predict_proba(X)    
    total_paid = sum(column_or_1d(y_pred))
    pert_paid = 100*total_paid/X.shape[0]
    print("Predicted Paid Claims = %s (%.1f%%)\n" %(total_paid, pert_paid))
    
    # output results
    proba = pd.DataFrame(data=probas[:,1], index = X.index, columns = ['Score'])
    yPred = pd.DataFrame(data=y_pred, index = X.index, columns = ['Pred'])
    res_pred = pd.concat([X, yPred, proba], axis=1)
    return res_pred 


def predict_with_cutoff(y_prob, df_y):
    n_events = df_y.values
    event_rate = sum(n_events) / float(df_y.shape[0]) * 100
    print(y_prob)
    threshold = np.percentile(y_prob[:, 1], 100 - event_rate)
    print ("Cutoff/threshold at: " + str(threshold))
    y_pred = [1 if x >= threshold else 0 for x in y_prob[:, 1]]
    return y_pred


def _replace_inf(X):
    # replace inf and na
    X_inf = X.replace(np.inf, np.nan)
   # X_na =X_inf.dropna(axis=1, how="all")    
    X = X_inf.fillna(method='ffill')
    X = X.fillna(method='pad')
    return X


def _report_class(classifier, name, xValid, yValid):
    # verify classifier model 
    #y_pred = classifier.predict(xValid)    
    y_pred = (classifier.predict(xValid.values) > 0.5).astype('int')
    
    classif_rate = np.mean(y_pred.ravel() == yValid.ravel()) * 100
    print("\nclassif_rate for %s : %f \n" % (name, classif_rate))

    test_results = ['Declined', 'Paid']   
    print ("\nModel Report")
    print(classification_report(yValid, y_pred, target_names= test_results))
    print("Predicted Paid Claims = %s" %sum(column_or_1d(y_pred)))
    print ("Accuracy : %.4g" % accuracy_score(yValid.values, y_pred))
    return y_pred

 
def _remove_Tags(Tags):                
    
    for item in DROPLIST:
        if item in Tags:
            Tags.remove(item)
    
    sensorTagsMu = ['mu_' + myitem for myitem in Tags]
    sensorTagsStd = ['std_' + myitem for myitem in Tags]
    sensorTagsMax = ['max_' + myitem for myitem in Tags]
    sensorTagsMin = ['min_' + myitem for myitem in Tags]
    sensorTagsMed = ['med_' + myitem for myitem in Tags]
#    sensorTagsVar = ['var_' + myitem for myitem in Tags]
    
    removeList = sensorTagsMu + sensorTagsStd + sensorTagsMax + sensorTagsMin + sensorTagsMed #+ sensorTagsVar
    Tags = remove_tag(Tags, removeList)
    
    #too scarce values
    sensorTagsM2d = ['m2d_' + myitem for myitem in TAGLIST_CON]
    Tags = remove_tag(Tags, sensorTagsM2d)
    
    #too large value for eng
    sensorTagsEng= ['eng_' + myitem for myitem in TAGLIST_CON]
    Tags = remove_tag(Tags, sensorTagsEng)
        
    #remove these two tags just for modifying the total number of features 600
    offTags = ['mu_CAC1_CARPET_SAG_AVE','max_CAC1_CARPET_SAG_AVE']
    Tags = remove_tag(Tags, offTags)
    
    #few values
    strTag = 'CAC1_Rear_Moisture_Meter'
    Tags = remove_strTag(Tags, strTag)

    #few values
    strTag = 'CAC1_Shearing_Moisture_Meter'
    Tags = remove_strTag(Tags, strTag)
    
    #few values
    strTag = 'CAC1_Gap_Height'
    Tags = remove_strTag(Tags, strTag)
    
    #no continuous
    strTag = 'CAC1_Current_Style'
    Tags = remove_strTag(Tags, strTag)
    
    #no data
    strTag = 'CAC1_Zone14_Bottom_Temp_Act'
    Tags = remove_strTag(Tags, strTag)
        
    strTag = 'lineSpeed'
    if strTag in Tags:
        Tags.remove(strTag)
    
    return Tags


#remove subset of tags from the main list
def remove_tag(Tags, removedList):
    for item in removedList:
        if item in Tags:
            Tags.remove(item)    
    return Tags 

def remove_space(DF, tag):
    #remove space from index in the given data frame   
    # data is a dat
    df_roll = [x.translate(dict.fromkeys(map(ord, whitespace))) if isinstance(x, str) else x for x in DF[tag].iloc[:]]
    DF[tag] = df_roll
    return DF


#remove a tag containing the string
def remove_strTag(Tags, strTag):    
    for item in Tags:
        if strTag in item:
            Tags.remove(item)    
    return Tags


def _verify_claim(classifier, name, xValid, yValid, test_results):
    # verify classifier model 
    #y_pred = classifier.predict(xValid)
    """
    The function cohen_kappa_score computes Cohen’s kappa statistic. 
    This measure is intended to compare labelings by different human 
    annotators, not a classifier versus a ground truth.

    The kappa score (see docstring) is a number between -1 and 1. Scores 
    above .8 are generally considered good agreement; zero or lower means 
    no agreement (practically random labels).
    """
    
    y_pred = (classifier.predict(xValid) > 0.5).astype('int')
    
    classif_rate = np.mean(y_pred.ravel() == yValid.ravel()) * 100
    print("\nclassify_rate for %s : %f \n" % (name, classif_rate))

    # Output probabilities
    #valFull = np.c_[xValid, yValid]
    probas = classifier.predict_proba(xValid)
    #y_pred = predict_with_cutoff(probas, yValid)  
    precision, recall, fscore, support = score(yValid, y_pred)
    
    accu = accuracy_score(yValid.values, y_pred)
    predEvent = sum(column_or_1d(y_pred))
    ck_score = cohen_kappa_score(yValid.values, y_pred)
    
    eval_res = np.append(recall, [predEvent, accu, ck_score])
        
    print("\nModel Report")
    print(classification_report(yValid, y_pred, target_names= test_results))
    print("\nPredicted Events = %s" % predEvent)
    print("Accuracy : %.4g" % accu)
    print("Cohen's kappa score : %.4g" % ck_score)
    
    # Plot normalized confusion matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(yValid, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes=test_results, normalize=True,
                           title='Normalized confusion matrix: '+name)
    plt.show()
        
    # gets a dictionary of {'class_name': probability}
#    prob_per_class = dict(zip(classifier.classes_, probas[1]))
#    print(prob_per_class)

    # output results
    proba = pd.DataFrame(data=probas[:,1], index = yValid.index, columns = ['Score'])
    yPred = pd.DataFrame(data=y_pred, index = yValid.index, columns = ['Pred'])
    res_pred = pd.concat([yValid, yPred, proba], axis=1)
    
    return res_pred, eval_res 