# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def multiclass_knn(X_train, X_test, y_train, y_test, kf_indices):
    
    print('Classification : KNN evaluation')
    
    ###############
    # Model cross-validation
    ###############
    neighbors = [3,4,5]
    
    model = Pipeline([('KNN', KNeighborsClassifier())])
    
    grid_search = GridSearchCV(model,
                        param_grid={'KNN__n_neighbors': neighbors},
                        cv=kf_indices,
                        scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                        n_jobs=2, return_train_score=True)
    

    grid_search.fit(X_train, y_train)
    
    
    print("Best hyperparameter(s) on the training set:", grid_search.best_params_)
    # print("Best estimator on the training set:", grid_search.best_estimator_)
    # print("Best score on the training set:", grid_search.best_score_)

    
    param_1_name = pd.DataFrame(grid_search.cv_results_['params']).iloc[:,0].name
    param_1_values = pd.DataFrame(grid_search.cv_results_['params']).iloc[:,0]
    cv_param = pd.DataFrame({param_1_name : param_1_values,
            'mean_test_score' : grid_search.cv_results_['mean_test_score'],
            'std_test_score' : grid_search.cv_results_['std_test_score'],
            'mean_train_score' : grid_search.cv_results_['mean_train_score'],
            'std_train_score' : grid_search.cv_results_['std_train_score']
            })
    
    plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
                 yerr=cv_param['std_test_score'])
    plt.errorbar(cv_param[param_1_name], cv_param['mean_train_score'],
                 yerr=cv_param['std_train_score'])
    # plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
    # plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
    plt.ylim(0,1.1)
    plt.ylabel("Accuracy criterion to maximize")
    plt.xlabel("Neighbors")
    plt.legend(['Cross validation set', 'Train set'])
    plt.title("KNN : assessing Train-CV performance over model complexity")
    plt.show()
    
    
    return grid_search

    

	



