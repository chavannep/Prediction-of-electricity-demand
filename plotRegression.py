import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_regression(model_classification, model_regression_0, model_regression_1,\
                    model_regression_2, X_test, y_test):
    

    ex_size = 500
    df_plot = {
        'location' : np.array(['west']*ex_size),
        'aver_temp_fall_winter' : np.random.uniform(X_test['aver_temp_fall_winter'].min(),X_test['aver_temp_fall_winter'].max(),size=ex_size),
        'aver_temp_spring_summer' : np.random.uniform(X_test['aver_temp_spring_summer'].min(),X_test['aver_temp_spring_summer'].max(),size=ex_size),
        'energ_indep' : np.array([0]*ex_size),
        'elec_demand' : np.array([0]*ex_size),
        'elec_demand_cluster' : np.array([-1]*ex_size),
        'elec_prediction' : np.array([0]*ex_size)
        }
    
    df_plot = pd.DataFrame.from_dict(df_plot)                   
    df_tmp = df_plot.iloc[:, ~df_plot.columns.isin(['location', 'elec_demand', 'elec_demand_cluster','elec_prediction','energ_indep'])]
    # Classification : which label fits the best to these inputs ?
    df_plot['elec_demand_cluster'] = model_classification.predict(df_tmp)
    
        
    ind_list_0 = df_plot.loc[df_plot['elec_demand_cluster'].isin([0])].index
    df_tmp= df_plot.loc[ind_list_0, ~df_plot.columns.isin(['location','elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
    df_plot['elec_prediction'].loc[ind_list_0] = model_regression_0.predict(df_tmp)
    
    ind_list_1 = df_plot.loc[df_plot['elec_demand_cluster'].isin([1])].index
    df_tmp= df_plot.loc[ind_list_1, ~df_plot.columns.isin(['location','elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
    df_plot['elec_prediction'].loc[ind_list_1] = model_regression_1.predict(df_tmp)
    
    ind_list_2 = df_plot.loc[df_plot['elec_demand_cluster'].isin([2])].index
    df_tmp= df_plot.loc[ind_list_2, ~df_plot.columns.isin(['location','elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
    df_plot['elec_prediction'].loc[ind_list_2] = model_regression_2.predict(df_tmp)
    
    
    fig, ax = plt.subplots()
    plt.scatter(df_plot['aver_temp_fall_winter'], df_plot['elec_prediction'], c='black')
    plt.scatter(X_test['aver_temp_fall_winter'], y_test, c='blue')
    plt.xlabel('Average temperature in fall-winter')
    plt.ylabel('Electricity demand')
    plt.legend(['Predicted data', 'Test data'])
    plt.title('Regression')
    plt.show()
