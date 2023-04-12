# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
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
    plt.ylabel("Accuracy score to maximize (-)")
    plt.xlabel("Neighbors")
    plt.legend(['Cross validation set', 'Train set'])
    plt.title("KNN validation curve : assessing Train-CV performance over model complexity")
    plt.show()
    
    #############
    # Learning curve : influence of the training set size
    #############

    # Compute the learning curve for a decision tree and vary the proportion of 
    # the training set from 10% to 100%.
    train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)

    results = learning_curve(
        grid_search.best_estimator_, X_train, y_train, train_sizes=train_sizes, cv=kf_indices,
        scoring='accuracy', n_jobs=2)
    train_size, train_scores, test_scores = results[:3]
    train_errors, test_errors = train_scores, test_scores
         
    fig, ax = plt.subplots()
    plt.errorbar(train_size, train_errors.mean(axis=1),
                 yerr=train_errors.std(axis=1), label="Train set error")
    plt.errorbar(train_size, test_errors.mean(axis=1),
                 yerr=test_errors.std(axis=1), label="CV set error")
    plt.legend()
    plt.ylim(0,1.1)
    plt.xscale("log")
    plt.xlabel("Number of samples in the training set")
    plt.ylabel("Accuracy score to maximize (-)")
    plt.title("KNN learning curve : assessing influence of training set size")
    plt.show()

    # Training error : if error very small, then the trained model is overfitting the training data.

    # Testing error alone : the more samples in training set, the lower the testing error. 
    # We are searching for the plateau of the testing error for which there is no benefit to adding samples anymore 

    # If already on a plateau and adding new samples in the training set does not reduce testing error, 
    # Bayes error rate may be reached using the available model. 
    # Using a more complex model might be the only possibility to reduce the testing error further.

    
    return grid_search

    

	



