# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def multiclass_bagging(X_train, X_test, y_train, y_test, kf_indices):
    
    print('Classification : bagging evaluation')
    
    ###############
    # Model 1 cross-validation (Decision tree base estimator)
    ###############
    max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_splits = [2, 10, 30, 50]
    min_leaves = [0.01, 0.05, 0.1, 1]
    param_distrib_DT={'decisionTreeClassifier__max_depth': max_depths,
                'decisionTreeClassifier__min_samples_split': min_splits,
                'decisionTreeClassifier__min_samples_leaf': min_leaves}

    model_DT = Pipeline([('decisionTreeClassifier', DecisionTreeClassifier(random_state=1))])

    grid_search_DT = GridSearchCV(model_DT,
                        param_grid=param_distrib_DT,
                        cv=kf_indices, scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                        n_jobs=2, return_train_score=True)

    # grid_search_DT = RandomizedSearchCV(model_DT,
    #                     param_distributions=param_distrib_DT,
    #                     cv=kf_indices, scoring='neg_mean_absolute_error', 
    #                     n_jobs=2, n_iter=20, random_state=1, return_train_score=True)

    grid_search_DT.fit(X_train, y_train)

    print("Best hyperparameter(s) on the training set:", grid_search_DT.best_params_)
    # print("Best estimator on the training set:", grid_search_DT.best_estimator_ )
    # print("Best score on the training set:", grid_search_DT.best_score_ )


    ###############
    # Model 2 cross-validation (Bagging trees estimator)
    ###############
    param_distrib_BT = {'baggingTreeClassifier__n_estimators': [1,10,20,50,100,200]}

    model_BT = Pipeline([
        ('baggingTreeClassifier', BaggingClassifier(base_estimator = grid_search_DT.best_estimator_,
                                                  verbose=0, random_state=1))])

    grid_search_BT = RandomizedSearchCV(model_BT,
                        param_distributions = param_distrib_BT,
                        cv=kf_indices, scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                        n_jobs=2, n_iter=20, random_state=1, return_train_score=True)


    grid_search_BT.fit(X_train, y_train)

    columns = [f"param_{name}" for name in param_distrib_BT.keys()]
    columns += ["mean_test_error", "std_test_error"]


    print("Best hyperparameter(s) on the training set:", grid_search_BT.best_params_)
    # print("Best estimator on the training set:", grid_search_BT.best_estimator_ )
    # print("Best score on the training set:", grid_search_BT.best_score_ )


    param_1_name = pd.DataFrame(grid_search_BT.cv_results_['params']).iloc[:,0].name
    param_1_values = pd.DataFrame(grid_search_BT.cv_results_['params']).iloc[:,0]
    cv_param = pd.DataFrame({param_1_name : param_1_values,
            'mean_test_score' : grid_search_BT.cv_results_['mean_test_score'],
            'std_test_score' : grid_search_BT.cv_results_['std_test_score'],
            'mean_train_score' : grid_search_BT.cv_results_['mean_train_score'],
            'std_train_score' : grid_search_BT.cv_results_['std_train_score']
            })
    
    plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
                 yerr=cv_param['std_test_score'])
    plt.errorbar(cv_param[param_1_name], cv_param['mean_train_score'],
                 yerr=cv_param['std_train_score'])
    # plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
    # plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
    plt.ylim(0,1.1)
    plt.ylabel("Accuracy criterion to maximize")
    plt.xlabel("Number of estimators (decision tree classifiers)")
    plt.legend(['Cross validation set', 'Train set'])
    plt.title("Bagging : assessing Train-CV performance over model complexity")
    plt.show()

        

    return grid_search_BT