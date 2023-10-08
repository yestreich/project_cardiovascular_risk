'''
==============
= Projekt ML =
==============
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # Das muss direkt nach dem Import stehen
from missingpy import MissForest #conda install --no-pin scikit-learn=1.1.2, downgrade notwendig
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from hyperopt import fmin, tpe, hp, Trials, space_eval, STATUS_OK
from sklearn.metrics import accuracy_score
#import random

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#%% Data

data = pd.read_csv("cardiovascular risk.csv")

print(data.columns)
print(data.head)
print(data.shape)

X = data.drop(["TenYearCHD", "id"], axis = 1)
y = np.asarray(data["TenYearCHD"])


#%% Missing Values

# NaN-Werte zählen
def miss_val_count(input):
    total = 0
    for i in input.columns:
        total += input[i].isna().sum()
        print(input[i].isna().sum(), ": ", i)
    print(total , ": total " , "(", round((total/input.shape[0])*100) , "%)")

miss_val_count(X)
# max 11% würden bei einfachem wegnehmen wegfallen
# würd ich gern behalten
# Denn BPMed ist ungleich besiedelt und könnte dann noch dünner werden.
# Deshalb (und zur Übung) wird alles befüllt

# Vorbereitung um alle NaN in 3 Schritten zu füllen : Kategorische Variablen in string und numerisch einteilen
s_cat = ["sex", "is_smoking"]
num_cat = ["education", "BPMeds"]

# Schritt 1 : String-Werte in numerische umgewandelt
enc = OrdinalEncoder()
X[s_cat] = enc.fit_transform(X[s_cat].values)

# Schritt 2 : numerisch-kategorische Variablen füllen (MissForest kann damit nicht umgehen) 
for cat in num_cat:
    uniques = X[cat].dropna().unique()
    value = np.random.choice(uniques, size = X[cat].isnull().sum(), replace=True)
    X.loc[X[cat].isnull(), cat] = value

# Schritt 3 : restliche NaN mit MissForest auffüllen
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

imputer = MissForest()
X_imp = imputer.fit_transform(X)

# Dataframe aus imputed values machen
X_imp = pd.DataFrame(data = X_imp, columns = X.columns)

# Check ob alle NaN weg sind
print(miss_val_count(X_imp))

#%% Outlier detection

for i in X.columns[1:]:
    plt.figure()
    sns.boxplot(X[i].values)
    plt.title(i)
    plt.show()
    
    plt.figure()
    sns.boxplot(X_imp[i].values)
    plt.title(i + "imp")
    plt.show()

# from pyod.models.abod import ABOD
# from pyod.models.cblof import CBLOF

# # ABOD
# outliers_fraction = 0.05
# abod_clf = ABOD(contamination = outliers_fraction)
# abod_clf.fit(X_imp[['BMI', 'glucose']])

# #Return the classified inlier/outlier
# abod_clf.labels_

# X_imp['ABOD_Clf'] = abod_clf.labels_

# g1 = sns.scatterplot(data = X_imp, x = 'BMI', y = 'glucose', hue = 'ABOD_Clf')
# plt.show(g1)

# #CBLOF

# random_state = 42
# cblof_clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state)
# cblof_clf.fit(X_imp[['BMI', 'glucose']])

# X_imp['CBLOF_Clf'] = cblof_clf.labels_
# g2 = sns.scatterplot(data = X_imp, x = 'BMI', y = 'glucose', hue = 'CBLOF_Clf')
# plt.show(g2)

#%% Split

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size = 0.2, stratify = y)
#y_train = y_train.reshape(y_train.shape[0], -1)

#%% Data balance

smt = SMOTENC([1,2,3,5,6,7,8], sampling_strategy = 'auto')
X_train, y_train = smt.fit_resample(X_train, y_train)

#Check ob Resample geklappt hat
print(pd.DataFrame(y_train).value_counts())

# Wie sind einst seltene Werte in X_train jetzt verteilt?
print(X_train["sex"].value_counts())
print(X_train["education"].value_counts())
print(X_train["is_smoking"].value_counts())
print(X_train["BPMeds"].value_counts())
print(X_train["prevalentStroke"].value_counts())
print(X_train["prevalentHyp"].value_counts())
print(X_train["diabetes"].value_counts())

#%% Standardization
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Ensemble - Bagging

# Bagging
model_bag = BaggingClassifier()
trained_bag = model_bag.fit(X_train_scaled, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_bag = cross_val_score(model_bag, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy Bagging CV: %.3f (%.3f)' % (np.mean(cv_bag), np.std(cv_bag)))
print("Accuracy Bagging: %.3f" % trained_bag.score(X_test_scaled, y_test))

#%% Ensemble - Stacking

#Base Learner
rfc = RandomForestClassifier()
xgb = XGBClassifier()
knn = KNeighborsClassifier()
#knn2 = KNeighborsClassifier(n_neighbors= 10)
#cbc = CatBoostClassifier()

base_learner = [("rfc", rfc),
                ("xgb", xgb),
                ("knn", knn),
#               ("knn2", knn2),
                #("cbc", cbc)
                ]

model_stack = StackingClassifier(estimators = base_learner, 
                          final_estimator = LogisticRegression())
trained_stack = model_stack.fit(X_train_scaled, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_stack = cross_val_score(model_stack, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy Stacking CV: %.3f (%.3f)' % (np.mean(cv_stack), np.std(cv_stack)))
print("Accuracy Stacking: %.3f" % trained_stack.score(X_test_scaled, y_test))


#%% Ensemble - Boosting

# Adaboost
model_ada = AdaBoostClassifier()
trained_ada = model_ada.fit(X_train_scaled,y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_ada = cross_val_score(model_ada, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy Ada CV: %.3f (%.3f)' % (np.mean(cv_ada), np.std(cv_ada)))
print("Accuracy Ada: %.3f" % trained_ada.score(X_test_scaled, y_test))

# XGBoost
model_xgb = XGBClassifier()
trained_xgb = model_xgb.fit(X_train_scaled, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_xgb = cross_val_score(model_xgb, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy XGB CV: %.3f (%.3f)' % (np.mean(cv_xgb), np.std(cv_xgb)))
print("Accuracy XGB: %.3f" % trained_xgb.score(X_test_scaled, y_test))

#CatBoost
model_cat = CatBoostClassifier()
trained_cat = model_cat.fit(X_train_scaled, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_cat = cross_val_score(model_cat, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy cat CV: %.3f (%.3f)' % (np.mean(cv_cat), np.std(cv_cat)))
print("Accuracy cat: %.3f" % trained_cat.score(X_test_scaled, y_test))

#GradientBoosting
model_grad = GradientBoostingClassifier()
trained_grad = model_grad.fit(X_train_scaled, y_train)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_grad = cross_val_score(model_grad, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Genauigkeit
print('Accuracy grad CV: %.3f (%.3f)' % (np.mean(cv_grad), np.std(cv_grad)))
print("Accuracy grad: %.3f" % trained_grad.score(X_test_scaled, y_test))


#%% Evaluierung des Defaults

print('Accuracy Bagging CV: %.3f (%.3f)' % (np.mean(cv_bag), np.std(cv_bag)))
print("Accuracy Bagging: %.3f" % trained_bag.score(X_test_scaled, y_test))
print('Accuracy Stacking CV: %.3f (%.3f)' % (np.mean(cv_stack), np.std(cv_stack)))
print("Accuracy Stacking: %.3f" % trained_stack.score(X_test_scaled, y_test))
print('Accuracy Ada CV: %.3f (%.3f)' % (np.mean(cv_ada), np.std(cv_ada)))
print("Accuracy Ada: %.3f" % trained_ada.score(X_test_scaled, y_test))
print('Accuracy XGB CV: %.3f (%.3f)' % (np.mean(cv_xgb), np.std(cv_xgb)))
print("Accuracy XGB: %.3f" % trained_xgb.score(X_test_scaled, y_test))
print('Accuracy cat CV: %.3f (%.3f)' % (np.mean(cv_cat), np.std(cv_cat)))
print("Accuracy cat: %.3f" % trained_cat.score(X_test_scaled, y_test))
print('Accuracy grad CV: %.3f (%.3f)' % (np.mean(cv_grad), np.std(cv_grad)))
print("Accuracy grad: %.3f" % trained_grad.score(X_test_scaled, y_test))

# Accuracy Bagging CV: 0.837 (0.015)
# Accuracy Bagging: 0.765
# Accuracy Stacking CV: 0.898 (0.013)
# Accuracy Stacking: 0.799
# Accuracy Ada CV: 0.736 (0.021)
# Accuracy Ada: 0.662
# Accuracy XGB CV: 0.894 (0.010)
# Accuracy XGB: 0.822
# Accuracy cat CV: 0.881 (0.012)
# Accuracy cat: 0.788
# Accuracy grad CV: 0.799 (0.019)
# Accuracy grad: 0.724
#%% Tuning von XGB

space_xgb = {
    "n_estimators": hp.choice("n_estimators", [25, 50, 100, 150, 200, 250]),
    "max_depth": hp.choice("max_depth", [1, 2, 3, 4, 5]),
    "learning_rate": hp.uniform("learning_rate", 0.05, 1.5),
    "subsample": hp.uniform("subsample", 0.0, 1.0),

}

def tuning_xgb(params):
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["learning_rate"] = float(params["learning_rate"])
    params["subsample"] = float(params["subsample"])

    # Initialize and train a classifier with the given hyperparameters
    clf_xgb = XGBClassifier(n_estimators = params['n_estimators'],
                                 max_depth = params['max_depth'],
                                 learning_rate = params['learning_rate'],
                                 subsample = params["subsample"], 
                                 )

    acc = cross_val_score(clf_xgb, X_train_scaled, y_train,scoring="accuracy").mean()
    return {"loss": -acc, "status":STATUS_OK}

# Initialize trials object
trials = Trials()

best_xgb = fmin(
    fn = tuning_xgb,
    space = space_xgb, 
    algo = tpe.suggest, 
    max_evals = 500, 
    trials = trials
)
#%% Getting the tuning values
print(space_eval(space_xgb, best_xgb)) # Keine Indexe, sondern direkt die echten Werte
best_loss = trials.best_trial['result']['loss']
print("Best loss:", best_loss)

# 100%|██████████| 500/500 [10:46<00:00,  1.29s/trial, best loss: -0.8728348654320378]
# {'learning_rate': 0.1981322926188378, 
# 'max_depth': 5, 
# 'n_estimators': 250, 
# 'subsample': 0.9538560173305269}

#%% Tuned XGB Model
clf_xgb = XGBClassifier(n_estimators = 250,
                              max_depth = 5,
                              learning_rate = 0.2,
                              subsample = 0.95 
                              )

#Training
clf_xgb.fit(X_train_scaled, y_train)
# Testing
y_pred_xgb = clf_xgb.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(accuracy_xgb) # 0.8141592920353983

#%%Tuning of Cat

space_cat = {
    'iterations': hp.choice('iterations', [400, 450, 500, 550, 600]),
    'leaf_estimation_iterations': hp.choice('leaf_estimation_iterations', [30, 50, 100]),
    'random_strength': hp.uniform('random_strength', 0.1, 1.0),
    'depth': hp.choice('depth', [2, 4, 6, 8, 10]),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.1, 5.0),
    'learning_rate': hp.uniform('learning_rate', 0.001, 1.0),
    'bagging_temperature': hp.uniform('bagging_temperature', 0.1, 1.0),
}

def tuning_cat(params):
    params["iterations"] = int(params["iterations"])
    params["leaf_estimation_iterations"] = int(params["leaf_estimation_iterations"])
    params["random_strength"] = float(params["random_strength"])
    params["depth"] = int(params["depth"])   
    params["l2_leaf_reg"] = float(params["l2_leaf_reg"])
    params["learning_rate"] = float(params["learning_rate"])
    params["bagging_temperature"] = float(params["bagging_temperature"])

    
    
    clf_cat = CatBoostClassifier(
        iterations = params['iterations'],
        leaf_estimation_iterations = params['leaf_estimation_iterations'],
        random_strength = params['random_strength'],
        depth = params['depth'],
        l2_leaf_reg = params['l2_leaf_reg'],
        learning_rate = params['learning_rate'],
        bagging_temperature = params['bagging_temperature'],
        task_type = "CPU"
    )

    acc = cross_val_score(clf_cat, X_train_scaled, y_train,scoring="accuracy").mean()
    return {"loss": -acc, "status":STATUS_OK}

#
trials_cat = Trials()

best_cat = fmin(
    fn = tuning_cat,
    space = space_cat, 
    algo = tpe.suggest, 
    max_evals = 100, 
    trials = trials_cat
)
#%% Getting the tuning values
print(space_eval(space_cat, best_cat)) # Keine Indexe, sondern direkt die echten Werte
best_loss = trials_cat.best_trial['result']['loss']
print("Best loss:", best_loss)

# 100%|██████████| 100/100 [1:09:41<00:00, 41.81s/trial, best loss: -0.8969061262750806]
# {'bagging_temperature': 0.839777932323215, 
# 'depth': 10, 
# 'iterations': 500, 
# 'l2_leaf_reg': 0.3882592777911337, 
# 'leaf_estimation_iterations': 100, 
# 'learning_rate': 0.03768945884017105, 
# 'random_strength': 0.9723015947414081}
#%% Tuned Cat_boost

cat_boost = CatBoostClassifier(
    iterations = 500,
    leaf_estimation_iterations = 100,
    random_strength = 0.9,
    depth = 10,
    l2_leaf_reg = 0.4,
    learning_rate = 0.04,
    bagging_temperature = 0.84,
    task_type = "CPU"
)

#Training
cat_boost.fit(X_train_scaled, y_train)
# Testing
y_pred_cat = cat_boost.predict(X_test_scaled)
accuracy_cat = accuracy_score(y_test, y_pred_cat)
print(accuracy_cat) # 0.8112094395280236

#%% Tuning of Stack

rfc = RandomForestClassifier()
xgb = XGBClassifier()
knn = KNeighborsClassifier()

space_rf = {'n_estimators': hp.choice('n_estimators_rf', [50, 75, 100, 125, 150]),
            'max_depth': hp.choice('max_depth_rf', [1, 2, 3, 4, 5]),
            'criterion': hp.choice('criterion_rf', ["gini", "entropy", "log_loss"])
            }

space_xgb = {"learning_rate" : hp.uniform("learning_rate_xgb", 0.1, 0.6),
             "max_depth" : hp.choice("max_depth_xgb", [1, 2, 3, 4, 5]), 
             "n_estimators" : hp.choice('n_estimators_xgb', [50, 75, 100, 125, 150])
             }

space_knn = {'n_neighbors': hp.choice('n_neighbors_knn', [2, 3, 4, 5, 6, 7, 8])
             }

space_clf = {
            'final_estimator': hp.choice('final_estimator_clf', ["LogisticRegression()",
                                                                 "RandomForestClassifier()",
                                                                 "XGBClassifier()"]),
            'cv_': hp.choice('cv_clf', [5, 10])
    }

def tuning_stack(params):   
    base_learner = [("rfc", rfc),
                  ("xgb", xgb),
                  ("knn", knn)]  
    
    stack_clf = StackingClassifier(estimators = base_learner,
                                   final_estimator = LogisticRegression())

    acc = cross_val_score(stack_clf, X_train_scaled, y_train,scoring="accuracy").mean()
    return {"loss": -acc, "status":STATUS_OK}

trials_stack = Trials()

space_stack = {"rf" : space_rf,
               "xgb" : space_xgb,
               "knn" : space_knn,
               "clf" : space_clf
               }

best_stack = fmin(
    fn = tuning_stack,
    space = space_stack, 
    algo = tpe.suggest, 
    max_evals = 100, 
    trials = trials_stack
)

#%% Getting the tuning values
print(space_eval(space_stack, best_stack)) 
best_loss = trials_stack.best_trial['result']['loss']
print("Best loss:", best_loss)

# 100%|██████████| 100/100 [27:45<00:00, 16.65s/trial, best loss: -0.8626271547714099]
# {'clf': {'cv_': 5, 
#          'final_estimator': 'LogisticRegression()'}, 
# 'knn': {'n_neighbors': 4}, 
# 'rf': {'criterion': 'entropy', 
#        'max_depth': 2, 
#        'n_estimators': 50}, 
# 'xgb': {'learning_rate': 0.141361094848536, 
#         'max_depth': 4, 
#         'n_estimators': 50}}

#%% Getuntes Stack -Modell
rfc = RandomForestClassifier(criterion = 'entropy', max_depth = 2, n_estimators = 50)
xgb = XGBClassifier(learning_rate = 0.14, max_depth = 4, n_estimators = 50)
knn = KNeighborsClassifier(n_neighbors = 4)

base_learner = [("rfc", rfc),
              ("xgb", xgb),
              ("knn", knn),
             #("knn2", knn2),
             #("cbc", cbc)
              ]  

stack_clf = StackingClassifier(estimators = base_learner,
                               final_estimator = LogisticRegression(), cv = 5)

#Traininig
stack_clf.fit(X_train_scaled, y_train)
#Testing
y_pred_stack = stack_clf.predict(X_test_scaled)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print(accuracy_stack) # 0.7625368731563422

#%% Voting Classifier

model_vote = VotingClassifier(estimators = [("stack_clf" ,stack_clf), 
                                            ("cat_boost", cat_boost), 
                                            ("clf_xgb", clf_xgb)],
                              )
# Cross-Validation
cv_vote = cross_val_score(model_vote, X_train_scaled, y_train,scoring="accuracy").mean()
print(cv_vote) #0.8799873286840437

# Training
model_vote = model_vote.fit(X_train_scaled, y_train)
#Testing
y_pred_vote = model_vote.predict(X_test_scaled)
accuracy_vote = accuracy_score(y_test, y_pred_vote)
print(accuracy_vote)   # 0.8112094395280236    


#%% Confusion Matrix - 

#XGBoost
confu_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_display = ConfusionMatrixDisplay(confu_xgb)
cm_display.plot()
plt.title("Confusion-Matrix XGBoost")
plt.savefig("Confusion_XGB")
plt.show()

#Catboost
confu_cat = confusion_matrix(y_test, y_pred_cat)
cm_display = ConfusionMatrixDisplay(confu_cat).plot()
plt.title("Confusion-Matrix Catboost")
plt.savefig("Confusion_Cat")
plt.show()

#Stack
confu_stack = confusion_matrix(y_test, y_pred_stack)
cm_display = ConfusionMatrixDisplay(confu_stack).plot()
plt.title("Confusion-Matrix Stacked Model")
plt.savefig("Confusion_Stack")
plt.show()

#Vote
confu_vote = confusion_matrix(y_test, y_pred_vote)
cm_display = ConfusionMatrixDisplay(confu_vote).plot()
plt.title("Confusion-Matrix Voting Model")
plt.savefig("Confusion_Vote")
plt.show()