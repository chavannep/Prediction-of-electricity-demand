# %pylab inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_validate, KFold, learning_curve
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
    
    # scores = pd.DataFrame(cv_results)
    # Not to store all parameters
    scores = pd.DataFrame()
    # scores['train_accuracy'] =  cv_results['train_accuracy']  
    # Correction for 'neg_mean_squared_error'
    scores['cv_error'] = - cv_results_0['test_score']
    scores['train_error'] = - cv_results_0['train_score']
       
    print(f"Mean squared error (macro) on CV set : "
          f"{scores['cv_error'].mean():.3f} ± {scores['cv_error'].std():.3f}")
       
       
    scores.plot.hist(bins=50, edgecolor="black")
    plt.xlabel("Mean squared error (-)")
    plt.title("Regressor 0 Linear cross validation : train and CV errors distribution")
    plt.show()
    # Small training error : the model is not under-fitting and is flexible enough to capture variations in the training set.
       
    # Larger testing error : the model is over-fitting : the model has memorized many variations of the training set 
    # that could be considered "noisy" because they do not generalize to help us make good prediction on the test set.            
       
       
    ##################
    # Training model
    ##################
    model_0.fit(X_train_0, y_train_0)
       
    #############
    # Learning curve : influence of the training set size
    #############
       
    # Compute the learning curve for a decision tree and vary the proportion of the training set from 10% to 100%.
    train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
       
    # Use a ShuffleSplit cross-validation to assess our predictive model.
    # cv = ShuffleSplit(n_splits=30, test_size=0.2)
       
    # Model or grid_search.best_estimator_
    results = learning_curve(
        model_0, X_train_0, y_train_0, train_sizes=train_sizes, cv=kf_indices_0,
        scoring='neg_mean_squared_error', n_jobs=2)
    train_size, train_scores, test_scores = results[:3]
    # Correction for 'neg_mean_squared_error'
    train_errors, test_errors = -train_scores, -test_scores
         
    fig, ax = plt.subplots()
    plt.errorbar(train_size, train_errors.mean(axis=1),
                 yerr=train_errors.std(axis=1), label="Train set error")
    plt.errorbar(train_size, test_errors.mean(axis=1),
                 yerr=test_errors.std(axis=1), label="CV set error")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Mean squared error (-)")
    plt.title("Regressor 0 Linear learning curve : assessing influence of training set size")
    plt.show()
       
    # Training error : if error very small, then the trained model is overfitting the training data.
       
    # Testing error alone : the more samples in training set, the lower the testing error. 
    # We are searching for the plateau of the testing error for which there is no benefit to adding samples anymore 
       
    # If already on a plateau and adding new samples in the training set does not reduce testing error, 
    # Bayes error rate may be reached using the available model. 
    # Using a more complex model might be the only possibility to reduce the testing error further.



    
    ind_list_1 = X_train.loc[X_train['elec_demand_cluster'].isin([1])].index
    X_train_1 = X_train.loc[ind_list_1, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_1 = y_train.loc[ind_list_1]
    kf_indices_1 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_1))
    model_1 = Pipeline([('linReg',  LinearRegression())])                            
    cv_results_1 = cross_validate(model_1, X_train_1, y_train_1, 
                                cv=kf_indices_1, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)
    # scores = pd.DataFrame(cv_results)
    # Not to store all parameters
    scores = pd.DataFrame()
    # scores['train_accuracy'] =  cv_results['train_accuracy']  
    # Correction for 'neg_mean_squared_error'
    scores['cv_error'] = - cv_results_1['test_score']
    scores['train_error'] = - cv_results_1['train_score']
       
    print(f"Mean squared error (macro) on CV set : "
          f"{scores['cv_error'].mean():.3f} ± {scores['cv_error'].std():.3f}")
       
       
    scores.plot.hist(bins=50, edgecolor="black")
    plt.xlabel("Mean squared error (-)")
    plt.title("Regressor 1 Linear cross validation : train and CV errors distribution")
    plt.show()

       
    ##################
    # Training model
    ##################
    model_1.fit(X_train_1, y_train_1)
       
    #############
    # Learning curve : influence of the training set size
    #############
       
    # Compute the learning curve for a decision tree and vary the proportion of the training set from 10% to 100%.
    train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
       
    # Use a ShuffleSplit cross-validation to assess our predictive model.
    # cv = ShuffleSplit(n_splits=30, test_size=0.2)
       
    # Model or grid_search.best_estimator_
    results = learning_curve(
        model_1, X_train_1, y_train_1, train_sizes=train_sizes, cv=kf_indices_1,
        scoring='neg_mean_squared_error', n_jobs=2)
    train_size, train_scores, test_scores = results[:3]
    # Correction for 'neg_mean_squared_error'
    train_errors, test_errors = -train_scores, -test_scores
         
    fig, ax = plt.subplots()
    plt.errorbar(train_size, train_errors.mean(axis=1),
                 yerr=train_errors.std(axis=1), label="Train set error")
    plt.errorbar(train_size, test_errors.mean(axis=1),
                 yerr=test_errors.std(axis=1), label="CV set error")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Mean squared error (-)")
    plt.title("Regressor 1 Linear learning curve : assessing influence of training set size")
    plt.show()



                                 
    
    ind_list_2 = X_train.loc[X_train['elec_demand_cluster'].isin([2])].index
    X_train_2 = X_train.loc[ind_list_2, ~X_train.columns.isin(['elec_demand_cluster'])]
    y_train_2 = y_train.loc[ind_list_2]
    kf_indices_2 = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train_2))
    model_2 = Pipeline([('linReg',  LinearRegression())])                            
    cv_results_2 = cross_validate(model_2, X_train_2, y_train_2, 
                                cv=kf_indices_2, scoring="neg_mean_squared_error",
                                return_train_score=True,
                                return_estimator=True)
    
    # scores = pd.DataFrame(cv_results)
    # Not to store all parameters
    scores = pd.DataFrame()
    # scores['train_accuracy'] =  cv_results['train_accuracy']  
    # Correction for 'neg_mean_squared_error'
    scores['cv_error'] = - cv_results_2['test_score']
    scores['train_error'] = - cv_results_2['train_score']
       
    print(f"Mean squared error (macro) on CV set : "
          f"{scores['cv_error'].mean():.3f} ± {scores['cv_error'].std():.3f}")
       
       
    scores.plot.hist(bins=50, edgecolor="black")
    plt.xlabel("Mean squared error (-)")
    plt.title("Regressor 2 Linear cross validation : train and CV errors distribution")
    plt.show()

       
    ##################
    # Training model
    ##################
    model_2.fit(X_train_2, y_train_2)
       
    #############
    # Learning curve : influence of the training set size
    #############
       
    # Compute the learning curve for a decision tree and vary the proportion of the training set from 10% to 100%.
    train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
       
    # Use a ShuffleSplit cross-validation to assess our predictive model.
    # cv = ShuffleSplit(n_splits=30, test_size=0.2)
       
    # Model or grid_search.best_estimator_
    results = learning_curve(
        model_2, X_train_2, y_train_2, train_sizes=train_sizes, cv=kf_indices_2,
        scoring='neg_mean_squared_error', n_jobs=2)
    train_size, train_scores, test_scores = results[:3]
    # Correction for 'neg_mean_squared_error'
    train_errors, test_errors = -train_scores, -test_scores
         
    fig, ax = plt.subplots()
    plt.errorbar(train_size, train_errors.mean(axis=1),
                 yerr=train_errors.std(axis=1), label="Train set error")
    plt.errorbar(train_size, test_errors.mean(axis=1),
                 yerr=test_errors.std(axis=1), label="CV set error")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Mean squared error (-)")
    plt.title("Regressor 2 Linear learning curve : assessing influence of training set size")
    plt.show()
       


    
    cv_error = -(len(ind_list_0)*cv_results_0["test_score"].mean() + \
                 len(ind_list_1)*cv_results_1["test_score"].mean() + \
                 len(ind_list_2)*cv_results_2["test_score"].mean() ) /\
        (len(ind_list_0) + len(ind_list_1) + len(ind_list_2))

    return model_0, model_1, model_2, cv_error
