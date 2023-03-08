# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def multiclass_random_forest(X_train, X_test, y_train, y_test, kf_indices):

    print('Classification : Random Forest evaluation')
    
    ###############
    # Model cross-validation
    ###############
    param_distrib = {'randomForestClassifier__max_features': [0.1, 0.2, 0.5, 0.8, 1.0],
                     'randomForestClassifier__n_estimators': [1,10,20,50,100,200]}

    model = Pipeline([('randomForestClassifier', RandomForestClassifier(n_jobs=2, verbose=0, random_state=1))
        ])

    rdm_search = RandomizedSearchCV(model,
                        param_distributions = param_distrib,
                        cv=kf_indices, scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                        n_jobs=2, n_iter=20, random_state=1, return_train_score=True)


    rdm_search.fit(X_train, y_train)


    print("Best hyperparameter(s) on the training set:", rdm_search.best_params_)
    # print("Best estimator on the training set:", rdm_search.best_estimator_ )
    # print("Best score on the training set:", rdm_search.best_score_ )
  
    
    param_1_name = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,0].name
    param_1_values = pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,0]
    cv_param = pd.DataFrame({param_1_name : param_1_values,
            'mean_test_score' : rdm_search.cv_results_['mean_test_score'],
            'std_test_score' : rdm_search.cv_results_['std_test_score'],
            'mean_train_score' : rdm_search.cv_results_['mean_train_score'],
            'std_train_score' : rdm_search.cv_results_['std_train_score']
            })

    plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
                 yerr=cv_param['std_test_score'])
    plt.errorbar(cv_param[param_1_name], cv_param['mean_train_score'],
                 yerr=cv_param['std_train_score'])
    # plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
    # plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
    plt.ylim(0,1.1)
    plt.ylabel("Accuracy criterion to maximize")
    plt.xlabel("Number of estimators")
    plt.legend(['Cross validation set', 'Train set'])
    plt.title("Random forest : assessing Train-CV performance over model complexity")
    plt.show()
    
    
    
    # ## Results from rdm_search search
    # results = rdm_search.cv_results_
    # params = {'randomForestClassifier__n_estimators': list(pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,0]),
    #           'randomForestClassifier__max_features': list(pd.DataFrame(rdm_search.cv_results_['params']).iloc[:,1])}
    # means_test = results['mean_test_score']
    # stds_test = results['std_test_score']
    # means_train = results['mean_train_score']
    # stds_train = results['std_train_score']
    # ## Getting indexes of values per hyper-parameter
    # masks=[]
    # masks_names= list(rdm_search.best_params_.keys())
    # for p_k, p_v in rdm_search.best_params_.items():
    #     masks.append(list(results['param_'+p_k].data==p_v))

    # # Cannot access explored parameters as a displayable full list
    # params = rdm_search.param_distributions    
    
    # ## Ploting results
    # fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    # fig.suptitle('Random forest : Train-CV comparison over model complexity')
    # fig.text(0.04, 0.5, 'Accuracy criterion to maximize', va='center', rotation='vertical')
    # pram_preformace_in_best = {}
    # for i, p in enumerate(masks_names):
    #     m = np.stack(masks[:i] + masks[i+1:])
    #     pram_preformace_in_best
    #     best_parms_mask = m.all(axis=0)
    #     best_index = np.where(best_parms_mask)[0]
    #     x = np.array(params[p])
    #     print(x)
    #     y_1 = np.array(means_test[best_index])
    #     print(np.where(best_parms_mask))
    #     e_1 = np.array(stds_test[best_index])
    #     y_2 = np.array(means_train[best_index])
    #     e_2 = np.array(stds_train[best_index])
    #     ax[i].errorbar(x, y_1, e_1, linestyle='-', marker='o', label='Cross-validation set', c='blue')
    #     ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='o',label='Train set', c='orange')
    #     ax[i].set_xlabel(p.upper())
    # plt.legend()
    # plt.show()
        
    return rdm_search