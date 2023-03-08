# %pylab inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def regression_regularized_polyfeat_ridge(X_train, y_train):

    ###############
    # Model cross-validation
    ###############

    alphas = np.logspace(-4, 1, num=20)
    degrees = np.array([1,2,3,4])
    model = Pipeline([('polyFeat',  PolynomialFeatures()),
                      ('ridge', Ridge())])                              
                                  

        
    ind_list_0 = X_train.loc[X_train['elec_demand_cluster'].isin([0])].index
    X_train_0= X_train.loc[ind_list_0, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_0 = y_train.loc[ind_list_0]
    kf_indices_0 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_0))
    model_0 = Pipeline([('polyFeat',  PolynomialFeatures()),
                      ('ridge', Ridge())])                              
    grid_search_0 = GridSearchCV(model_0,
                        param_grid={'ridge__alpha': alphas,
                                    'polyFeat__degree': degrees},
                        cv=kf_indices_0,
                        scoring="neg_mean_squared_error", 
                        n_jobs=2, return_train_score=True)
    grid_search_0.fit(X_train_0, y_train_0)
 
    # print("Best hyperparameter(s) on the training set:", grid_search_0.best_params_)
    # print("Best estimator on the training set:", grid_search_0.best_estimator_)
    # print("Best score on the training set:", grid_search_0.best_score_)

    ind_list_1 = X_train.loc[X_train['elec_demand_cluster'].isin([1])].index
    X_train_1= X_train.loc[ind_list_1, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_1 = y_train.loc[ind_list_1]
    kf_indices_1 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_1))
    model_1 = Pipeline([('polyFeat',  PolynomialFeatures()),
                      ('ridge', Ridge())])                              
    grid_search_1 = GridSearchCV(model_1,
                        param_grid={'ridge__alpha': alphas,
                                    'polyFeat__degree': degrees},
                        cv=kf_indices_1,
                        scoring="neg_mean_squared_error", 
                        n_jobs=2, return_train_score=True)
    grid_search_1.fit(X_train_1, y_train_1)

    # print("Best hyperparameter(s) on the training set:", grid_search_1.best_params_)
    # print("Best estimator on the training set:", grid_search_1.best_estimator_)
    # print("Best score on the training set:", grid_search_1.best_score_)
    
 
    ind_list_2 = X_train.loc[X_train['elec_demand_cluster'].isin([2])].index
    X_train_2= X_train.loc[ind_list_2, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_2 = y_train.loc[ind_list_2]
    kf_indices_2 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_2))
    model_2 = Pipeline([('polyFeat',  PolynomialFeatures()),
                      ('ridge', Ridge())])                              
    grid_search_2 = GridSearchCV(model_2,
                        param_grid={'ridge__alpha': alphas,
                                    'polyFeat__degree': degrees},
                        cv=kf_indices_2,
                        scoring="neg_mean_squared_error", 
                        n_jobs=2, return_train_score=True)
    grid_search_2.fit(X_train_2, y_train_2)   

    # print("Best hyperparameter(s) on the training set:", grid_search_2.best_params_)
    # print("Best estimator on the training set:", grid_search_2.best_estimator_)
    # print("Best score on the training set:", grid_search_2.best_score_)


                 
    ## Results from grid_search_0 search
    results = grid_search_0.cv_results_

    means_test = np.multiply(-1,results['mean_test_score'])
    stds_test = results['std_test_score']
    means_train = np.multiply(-1,results['mean_train_score'])
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid_search_0.best_params_.keys())
    for p_k, p_v in grid_search_0.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params = grid_search_0.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Polyfeat - Regularized - Ridge : Train-CV comparison over model 0 complexity')
    fig.text(0.04, 0.5, 'Mean squared error to minimize', va='center', rotation='vertical')
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
    
    
    
    
    ## Results from grid_search_1 search
    results = grid_search_1.cv_results_
    
    means_test = np.multiply(-1,results['mean_test_score'])
    stds_test = results['std_test_score']
    means_train = np.multiply(-1,results['mean_train_score'])
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid_search_1.best_params_.keys())
    for p_k, p_v in grid_search_1.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params = grid_search_1.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Polyfeat - Regularized - Ridge : Train-CV comparison over model 1 complexity')
    fig.text(0.04, 0.5, 'Mean squared error to minimize', va='center', rotation='vertical')
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
    
    
    
    ## Results from grid_search_2 search
    results = grid_search_2.cv_results_
   
    means_test = np.multiply(-1,results['mean_test_score'])
    stds_test = results['std_test_score']
    means_train = np.multiply(-1,results['mean_train_score'])
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid_search_2.best_params_.keys())
    for p_k, p_v in grid_search_2.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params = grid_search_2.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Polyfeat - Regularized - Ridge : Train-CV comparison over model 2 complexity')
    fig.text(0.04, 0.5, 'Mean squared error to minimize', va='center', rotation='vertical')
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
    
    
    cv_error = -(len(ind_list_0)*grid_search_0.best_score_ + \
                 len(ind_list_1)*grid_search_1.best_score_ + \
                 len(ind_list_2)*grid_search_2.best_score_ ) /\
        (len(ind_list_0) + len(ind_list_1) + len(ind_list_2))
    
    
    return grid_search_0, grid_search_1, grid_search_2, cv_error
