# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def multiclass_svm(X_train, X_test, y_train, y_test, kf_indices):

    print('Classification : SVM evaluation')
    
    ###############
    # Model cross-validation
    ###############

    c_values = np.logspace(-4, 1, num=6)
    gammas = np.logspace(-4, 1, num=6)
    degrees = [2,3,4,5,6]

    model = Pipeline([('SVC', SVC(kernel="rbf", random_state=1))])
    # LinearSVC(C=C, max_iter=10000)
    grid_search = GridSearchCV(model,
                        param_grid={'SVC__C': c_values,
                                    'SVC__gamma': gammas, 
                                    'SVC__degree': degrees},
                        cv=kf_indices,
                        scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                        n_jobs=2, return_train_score=True)

    grid_search.fit(X_train, y_train)

    print("Best hyperparameter(s) on the training set:", grid_search.best_params_)
    # print("Best estimator on the training set:", grid_search.best_estimator_ )
    # print("Best score on the training set:", grid_search.best_score_ )
    

    ## Results from grid_search search
    results = grid_search.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid_search.best_params_.keys())
    for p_k, p_v in grid_search.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params = grid_search.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('SVM : Train-CV comparison over model complexity')
    fig.text(0.04, 0.5, 'Accuracy criterion to maximize', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='-', marker='o', label='Cross-validation set', c='blue')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='o',label='Train set', c='orange')
        ax[i].set_xlabel(p.upper())
    plt.legend()
    plt.show()

    
    return grid_search



	



