# %pylab inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline


def regression_simple_linear(X_train, y_train):

    ###############
    # Model cross-validation
    ###############

    # train_error = -cv_results["train_score"].mean()
    # cv_error = -cv_results["test_score"].mean()


    ind_list_0 = X_train.loc[X_train['elec_demand_cluster'].isin([0])].index
    X_train_0 = X_train.loc[ind_list_0, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_0 = y_train.loc[ind_list_0]
    kf_indices_0 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_0))
    model_0 = Pipeline([('linReg',  LinearRegression())])                            
    cv_results_0 = cross_validate(model_0, X_train_0, y_train_0, 
                                cv=kf_indices_0, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)
    model_0.fit(X_train_0, y_train_0)
 
    
    ind_list_1 = X_train.loc[X_train['elec_demand_cluster'].isin([1])].index
    X_train_1 = X_train.loc[ind_list_1, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_1 = y_train.loc[ind_list_1]
    kf_indices_1 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_1))
    model_1 = Pipeline([('linReg',  LinearRegression())])                            
    cv_results_1 = cross_validate(model_1, X_train_1, y_train_1, 
                                cv=kf_indices_1, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)
    model_1.fit(X_train_1, y_train_1)
                                 
    
    ind_list_2 = X_train.loc[X_train['elec_demand_cluster'].isin([2])].index
    X_train_2 = X_train.loc[ind_list_2, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_2 = y_train.loc[ind_list_2]
    kf_indices_2 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_2))
    model_2 = Pipeline([('linReg',  LinearRegression())])                            
    cv_results_2 = cross_validate(model_2, X_train_2, y_train_2, 
                                cv=kf_indices_2, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)
    model_2.fit(X_train_2, y_train_2)
    
    cv_error = -(len(ind_list_0)*cv_results_0["test_score"].mean() + \
                 len(ind_list_1)*cv_results_1["test_score"].mean() + \
                 len(ind_list_2)*cv_results_2["test_score"].mean() ) /\
        (len(ind_list_0) + len(ind_list_1) + len(ind_list_2))

    return model_0, model_1, model_2, cv_error
