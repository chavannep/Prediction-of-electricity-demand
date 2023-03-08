# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def multiclass_gbdt(X_train, X_test, y_train, y_test, kf_indices):
    
    print('Classification : GBDT evaluation')  

    ###############
    # Model cross-validation
    ###############
    param_distrib = {'gradBoostClassifier__n_estimators': [1, 10, 50, 100, 200, 300, 400, 500], 
                           'gradBoostClassifier__max_leaf_nodes': [2, 5, 10, 20, 50, 100],
                           'gradBoostClassifier__max_depth': [2, 5, 8, 10],
                           'gradBoostClassifier__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]}
    model = Pipeline([('gradBoostClassifier', GradientBoostingClassifier(verbose=0, random_state=1))])

    rdm_search = RandomizedSearchCV(model,
                        param_distributions=param_distrib,
                        cv=kf_indices, scoring='accuracy', # 'f1' if binary, 'accuracy' if multi 
                        n_jobs=2, n_iter=20, random_state=1, return_train_score=True)

    ##################
    # Training model
    ##################
    rdm_search.fit(X_train, y_train)

    print("Best hyperparameter(s) on the training set:", rdm_search.best_params_)
    # print("Best estimator on the training set:", rdm_search.best_estimator_ )
    # print("Best score on the training set:", rdm_search.best_score_ )


    # Weird display concerning RandomizedSearchCV
    
    
    
    # param_1_name = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,0].name
    # param_1_values = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,0]
    # cv_param = pd.DataFrame({param_1_name : param_1_values,
    #         'mean_test_score' : rdm_search.cv_results_['mean_test_score'],
    #         'std_test_score' : rdm_search.cv_results_['std_test_score'],
    #         'mean_train_score' : rdm_search.cv_results_['mean_train_score'],
    #         'std_train_score' : rdm_search.cv_results_['std_train_score']
    #         })

    # plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
    #              yerr=cv_param['std_test_score'])
    # plt.errorbar(cv_param[param_1_name], cv_param['mean_train_score'],
    #              yerr=cv_param['std_train_score'])
    # # plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
    # # plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
    # plt.ylim(0,1.1)
    # plt.ylabel("Accuracy criterion to maximize")
    # plt.xlabel("Number of estimators")
    # plt.legend(['Cross validation set', 'Train set'])
    # plt.title("GBDT : assessing Train-CV performance over model complexity")
    # plt.show()
    
    
    # param_1_name = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,3].name
    # param_1_values = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,3]
    # cv_param = pd.DataFrame({param_1_name : param_1_values,
    #         'mean_test_score' : rdm_search.cv_results_['mean_test_score'],
    #         'std_test_score' : rdm_search.cv_results_['std_test_score'],
    #         'mean_train_score' : rdm_search.cv_results_['mean_train_score'],
    #         'std_train_score' : rdm_search.cv_results_['std_train_score']
    #         })

    # plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
    #              yerr=cv_param['std_test_score'])
    # plt.errorbar(cv_param[param_1_name], cv_param['mean_train_score'],
    #              yerr=cv_param['std_train_score'])
    # # plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
    # # plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
    # plt.ylim(0,1.1)
    # plt.ylabel("Accuracy criterion to maximize")
    # plt.xlabel("Learning rate")
    # plt.legend(['Cross validation set', 'Train set'])
    # plt.title("GBDT : assessing Train-CV performance over model complexity")
    # plt.show()
    
    return rdm_search