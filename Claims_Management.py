# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:30:30 2020

@author: GAURAV SHARMA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn import feature_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def analysefeatures(pdf):
    inttypes=['int16','int32','int64']
    floattypes=['float32','float64']
    intcols=pdf.select_dtypes(include=inttypes).columns.values
    floatcols=pdf.select_dtypes(include=floattypes).columns.values
    catcols=pdf.select_dtypes(include=['object']).columns.values
    return intcols,floatcols,catcols

def modelwisetest(modellist,Xtrain,Xtest,Ytrain,Ytest):
    serieslist=[]
    for model in modellist:
        scores=np.zeros(Xtrain.shape[1])
        for i,col in enumerate(Xtrain.columns.values):
             #model=ensemble.RandomForestClassifier(100)
             
             model.fit(Xtrain[col].values.reshape(-1,1),Ytrain)
             predtest=model.predict(Xtest[col].values.reshape(-1,1))
             auc=metrics.roc_auc_score(Ytest,predtest)
             scores[i]=auc
        ser=pd.Series(scores)
        ser.index=Xtrain.columns
        serieslist.append(ser.sort_values(ascending=False))
    return serieslist

def reports(ytrue,predicted):
    print("Accuracy : {}".format(metrics.accuracy_score(ytrue,predicted)))
    print("Precision : {}".format(metrics.precision_score(ytrue,predicted)))
    print("Recall : {}".format(metrics.recall_score(ytrue,predicted)))
    print("F1_score : {}".format(metrics.f1_score(ytrue,predicted)))
    ##print("Logloss : {}".format(metrics.log_loss(ytrue,predicted)))
    print("AUC : {}".format(metrics.roc_auc_score(ytrue,predicted)))

def build(Xtrain,Xtest,ytrain,ytest):
    model=linear_model.LogisticRegression()
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)
    
def build1(Xtrain,Xtest,ytrain,ytest):
    model=ensemble.RandomForestClassifier(n_estimators=400,max_depth=20,min_samples_split=18,min_samples_leaf=9,random_state=5)
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)
    
    
def build2(Xtrain,Xtest,ytrain,ytest):
    model=tree.DecisionTreeClassifier(max_depth=22,min_samples_split=10,min_samples_leaf=15,random_state=42)
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)    
    
#build1(Xtrain_all,Xtest_all,Ytrain_all,Ytest_all)    
def modelstats1(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.f1_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.f1_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","F1","AUC"]
    return pd.DataFrame(stats,columns=colnames)

df=pd.read_csv('D:\\ML_DATA\\claims_management\\train.csv')
df.drop('ID',inplace=True,axis=1)
df.shape
df.head()
intcols,floatcols,catcols=analysefeatures(df)
######### cat_df #############
catcols
cat_df=pd.DataFrame(df[catcols])
cat_df=pd.concat([cat_df,df.target],axis=1)

########## int_df ###############
intcols
int_df=pd.DataFrame(df[intcols])

######### float_df ################
floatcols
float_df=pd.DataFrame(df[floatcols])
float_df=pd.concat([float_df,df.target],axis=1)
#float_df.fillna(float_df.mean())
imputer=KNNImputer( n_neighbors=6)

mydf=pd.DataFrame()
size=float_df.shape[0]

i=0
while i < size:
    mydf=mydf.append(pd.DataFrame(imputer.fit_transform(float_df[i:i+2000])))
    i+=2000

X_f=mydf.drop(108,axis=1)
Y_f=mydf[108]
Xtrain_f,Xtest_f,Ytrain_f,Ytest_f=model_selection.train_test_split(X_f,Y_f,test_size=.2,random_state=0)
fvalue,probability=feature_selection.f_classif(Xtrain_f,Ytrain_f)
ser=pd.Series(probability)
ser.index=Xtrain_f.columns
ser[:20].sort_values(ascending=False).plot.bar()
ser.sort_values(ascending=False,inplace=True)
ser.plot.bar(rot=0)
ser1=pd.Series(fvalue)
ser1.index=Xtrain_f.columns
ser1[:20].sort_values(ascending=False).plot.bar()
ser1.sort_values(ascending=False,inplace=True)
obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=15) 
obj.fit(Xtrain_f,Ytrain_f)

obj.fit(X_f,Y_f)
X_f.columns.values[obj.get_support()]

Xtrain_f.columns.values[obj.get_support()]
#array([  2,   8,  12,  19,  28,  40,  42,  53,  68,  75,  83,  88,  92,
#        97, 101]


Xtrain_anova=obj.transform(Xtrain_f)
Xtrain_anova=pd.DataFrame(Xtrain_anova)
Xtrain_anova.columns=[  2,   8,  12,  19,  28,  40,  42,  53,  68,  75,  83,  88,  92,97, 101]
        
Xtest_anova=obj.transform(Xtest_f)
Xtest_anova=pd.DataFrame(Xtest_anova)
Xtest_anova.columns=[  2,   8,  12,  19,  28,  40,  42,  53,  68,  75,  83,  88,  92,
        97, 101]
lst_anova=[  2,   8,  12,  19,  28,  40,  42,  53,  68,  75,  83,  88,  92,
        97, 101]
arr_anova=np.array(lst_anova)
############### Xtrain_anova and Xtest_anova
Xtrain_anova=Xtrain_f[arr_anova]
Xtest_anova=Xtest_f[arr_anova]
list=[Ytrain_f,Xtrain_anova]
f_mydf=pd.concat(list,axis=1)
############## Making ddf from X_f 

X_f1=X_f[lst_anova]
X_f1
############## Now processing categorical columns ###########
fcat_df=cat_df.drop(['v3','v30','v31','v113','v125','target','v22','v56','v52','v91','v107','v112'],axis=1)
fcat_df.shape

enc_fdf=fcat_df.apply(LabelEncoder().fit_transform)

fint_df=int_df.drop(['target'],axis=1)
fint_df.shape       


#########################################3

# feature selection
def select_features(X_train, y_train):
	fs = SelectKBest(score_func=chi2, k=4)
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
#	X_test_fs = fs.transform(X_test)
	return X_train_fs

X_train_fs = select_features(fcat_df,Y_f)



# Categorical boolean mask
categorical_feature_mask = fcat_df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = fcat_df.columns[categorical_feature_mask].tolist()
categorical_cols

label_encoder=preprocessing.LabelEncoder()
fcat_df.columns.unique
for i in fcat_df.columns:
    fcat_df[i]=label_encoder.fit_transform(df[i])
    









objmi = feature_selection.SelectKBest(score_func=feature_selection.chi2
                                      ,k='2')
#applying murtual_info_classif :
objmi = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_classif
                                      ,k='4')
objmi.fit(fint_df,Y_f)
# X_cat_train.columns.values[objmi.get_support()]
# bar plot on mi scores :
ser_mi=objmi.scores_
ser_mi=pd.Series(ser_mi)
# ser_mi.index=X_cat_train.columns
ser_mi.sort_values(ascending=False).plot.bar()

X_f1.reset_index(drop=True,inplace=True)

     
fin_df=pd.concat([fint_df,fcat_df,X_f1],axis=1)




fin_df=fin_df.rename( columns={2:"c1", 8:"c2", 12:"c3", 19:"c4", 28:"c5", 42:"c6", 92:"c7", 97:"c8"})
      

obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=15) 
obj.fit(fin_df,Y_f)
fin_df.columns.values[obj.get_support()]
      

lst_anova_rfe=['v38', 'v62', 'v72', 'v129', 'v47', 'v66', 'v110', 'c1', 'c2','c3', 'c4', 'c5', 'c6', 'c7', 'c8']
       
arr_anova_rfe=np.array(lst_anova_rfe)

X_final=fin_df[arr_anova_rfe]




fvalue,probability=feature_selection.f_classif(X_final,Y_f)
ser=pd.Series(probability)
ser.index=X_final.columns
ser[:13].sort_values(ascending=False).plot.bar()
ser.sort_values(ascending=False,inplace=True)
ser.plot.bar(rot=0)
ser1=pd.Series(fvalue)
ser1.index=X_final.columns
ser1[:13].sort_values(ascending=False).plot.bar()
ser1.sort_values(ascending=False,inplace=True)
obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=13) 
obj.fit(X_final,Y_f)


X_final.columns.values[obj.get_support()]

Xtrain_f,Xtest_f,Ytrain_f,Ytest_f=model_selection.train_test_split(X_final,Y_f,test_size=.28,random_state=0)
build1(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)
build(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)
modelstats1(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)
#build1(Xtrain_all,Xtest_all,Ytrain_all,Ytest_all)

build2(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)


f, ax = plt.subplots(figsize=(10, 6))
corr = np.corrcoef(X_final)
sns.heatmap(X_final, annot=True, cmap="coolwarm",fmt='.2f',linewidths=.05)
                 
f.subplots_adjust(top=0.93)
t= f.suptitle('Claim Attributes Correlation Heatmap', fontsize=14)
