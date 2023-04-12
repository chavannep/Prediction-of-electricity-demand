###############
# Preliminary overview
###############

# A representative dataset has been collected in terms of electricity demand from consumers and other company departments.
# The company wants to use these data in order to build a forecasting model for electricity supply.
# Features of interest (identified after data analysis) collected among the dataset are : 
#     - Average temperature in fall-winter : the colder the weather, the higher the electricity demand in terms of heating
#     - Average temperature in spring-summer : the hotter the weather, the higher the electricity in terms of cooling
#     - Location : local weather emphasizes consuming trends by increasing/decreasing
#     - Energy independence : if consumers have other sources of energy than just electricity, so this last demand will be lower
#     - Electricity consumption : this is the target which should be found with the final model. 
#     However it has been collected in this dataset to build the model. 
#     But it will not be available in the future as a feature.
    
# Several steps will be performed before getting the final model :
#     - Clustering : find clusters of electricity demand among the unlabeled dataset of consumers
#     - Classification : build multiclassification model to label data (current and future ones)
#     - Regression : build regression model for each labeled cluster of consumers in order to forecast electricity demand 
#     For those steps, best model will be used over several algorithms (KNN, SVM, regularized, poly features...)
#     - Prediction : perform clustering, classification and regression steps for raw data (previous features excluding electricity consumption)
#     - Final model : calculate electricity supply which should be 10% higher than computed forecast electricity demand (to jump over margin of error)
#     electricity supply = 1.1 * electriciy forecast demand 
#                        = 1.1 * (pop_cluster_1*elec_demand_cluster_1 + 
#                                 pop_cluster_2*elec_demand_cluster_2 + ...)

# Disclaimer : 
#    - raw data have been created by my own and are not real.  
#    - this forecast model is just a very simple application of un/supervised learning algos for clustering, classification and regression in a single project.
#    - cluster population = nb rows in dataset are not very high in order to be quickly computable on my laptop.
#    - this is more a proof of concept than a real robust model : several improvements should be made
#    for instance for regression step (new data are computed with regression model fitted on main dataset which does not contain this new data...)
#    and for data declaration : comprehensive coding has been avoided when considering clusters
#    it is easier to read this code for 3 clusters but it should have been more automatic for larger clusters.

#!/usr/bin/python


from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



# Clustering subroutine
from clusteringKMeans import clustering_kmeans
from plotDecisionBoundaryClustering import plot_decision_boundary_clustering

# Classification subroutines
from multiClassKNN import multiclass_knn
from multiClassSVM import multiclass_svm
from plotDecisionBoundaryClassification import plot_decision_boundary_classification

# Regression subroutines
from regressionSimpleLinear import regression_simple_linear
from regressionRegPolyRidge import regression_regularized_polyfeat_ridge
from plotRegression import plot_regression

###########################################
# Data analysis
###########################################

###############
# Loading data
###############
df = pd.read_csv('./dataSetElec.csv', delimiter=',', index_col=0)

# Encoding 'location' = 1st column
encoder = LabelEncoder()
df.iloc[:,0] = encoder.fit_transform(df.iloc[:,0])

# Adding future computed parameters to df
df['elec_demand_cluster'] = -1
df['elec_prediction'] = -1

# Choosing two features to display data
fig, ax = plt.subplots()
ax.scatter(df['aver_temp_fall_winter'], df['elec_demand'], marker="o", color="k")
plt.xlabel('Average temperature in fall-winter (°C)')
plt.ylabel('Electricity demand (kWh)')
plt.title('Overview of raw data')
plt.show()


# Normalizing data
df_tmp = df[df.columns.difference(['location', 'elec_demand_cluster', 'elec_demand', 'elec_prediction'])]
scaler_data = StandardScaler().fit(df_tmp)

df_tmp = df['elec_demand']
scaler_target = StandardScaler().fit(df_tmp.values.reshape(-1, 1))

df_tmp = df[df.columns.difference(['location', 'elec_demand_cluster', 'elec_demand', 'elec_prediction'])]
df_tmp_1 = pd.DataFrame(scaler_data.transform(df_tmp), columns=df_tmp.columns)

df_tmp = pd.DataFrame(scaler_target.transform(df['elec_demand'].values.reshape(-1, 1)), columns=[df['elec_demand'].name]) 

df_tmp = pd.concat([df_tmp_1, df_tmp], join = 'outer', axis = 1)
df_tmp_1 = df[['location', 'elec_demand_cluster', 'elec_prediction']]
df = pd.concat([df_tmp, df_tmp_1], join = 'outer', axis = 1)




################
# Data checking
################
data = df[df.columns.difference(['elec_demand', 'elec_demand_cluster', 'elec_prediction'])]


# Step 1 multicolinearity : VIF score
vif_df = pd.DataFrame(data={'Feature' : data.columns, 
            'VIF_score' : [variance_inflation_factor(data.values, i) \
                                      for i in range(data.shape[1])]}) 
    
data = df[df.columns.difference(['elec_demand_cluster', 'elec_prediction'])]
# Step 2 multicolinearity : correlation matrix
# Pair plot
sns.pairplot(data)
plt.show()

# Correlation matrix
corr = data.corr()
f, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap, vmin=0, vmax=1)
plt.show()





###########################################
# Clustering 
###########################################
# Select two features after which clusters are clearly identified
model_clustering, df['elec_demand_cluster'], nb_clusters, score_kmeans, centroids\
    = clustering_kmeans(df[['aver_temp_fall_winter', 'elec_demand']])
print('\nClustering : nb of clusters :', nb_clusters)
print("\nClustering : silhouette score : %.3f" \
      % silhouette_score(df[['aver_temp_fall_winter', 'elec_demand']], df['elec_demand_cluster'], metric="euclidean"))

plot_decision_boundary_clustering(model_clustering, df['aver_temp_fall_winter'],\
                                  df['elec_demand'], df['elec_demand_cluster'],\
                                      centroids.iloc[:,0], centroids.iloc[:,1])

print('\nClustering : end of section : press a key to continue')
input('')








###########################################
# Multi-classification
###########################################
# Classification : target is elec_demand_cluster as a label
data = df[df.columns.difference(['location', 'elec_demand', 'elec_demand_cluster', 'elec_prediction','energ_indep'])]
target = df['elec_demand_cluster']

################
# Splitting into train-test sets
################
# Normalization occurs later in pipelines (StandardScaler(),...)
X_train, X_test, y_train, y_test = train_test_split( 
    data, target, test_size=0.3, shuffle=True, random_state=1)

###############
# Splitting previous training data into k-1 sets (2nd training) and one CV set
############### 
kf_indices = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X_train))


# Storing accuracy score for each model (optimized separaterly through a grid/random search)
multiclass_dict= dict()
model_knn = multiclass_knn(X_train, X_test, y_train, y_test, kf_indices)
model_svm = multiclass_svm(X_train, X_test, y_train, y_test, kf_indices)
multiclass_dict['KNN'] = (model_knn.best_estimator_, model_knn.best_score_)
multiclass_dict['SVM'] = (model_svm.best_estimator_, model_svm.best_score_)

fig, ax = plt.subplots()
plt.bar(x=list(multiclass_dict.keys()), height=[list(list(multiclass_dict.values())[0])[1],\
                                                list(list(multiclass_dict.values())[1])[1]],\
        color='g')
plt.xticks(rotation=45)
plt.xlabel('Multi-classification models')
plt.ylabel('Accuracy score (-)')
plt.title('Comparing multi-class models performance')
plt.show()


# Model chosen after max accuracy score
model_classification = max(multiclass_dict.values(), key=lambda sub: sub[1])[0]
best_score = max(multiclass_dict.values(), key=lambda sub: sub[1])[1]
# model_name = [key for key, val in multiclass_dict.items() if val == best_score][0]

plot_decision_boundary_classification(model_classification,\
                                      X_test['aver_temp_fall_winter'],\
                                          X_test['aver_temp_spring_summer'],\
                                              y_test)

print('\nClassification : model characteristics :',  model_classification)
print('\nClassification : best score :',  best_score)

y_pred = model_classification.predict(X_test) 

print("\nClassification : report on the test set:\n%str" % classification_report(y_test, y_pred))
print('\nClassification : end of section : press a key to continue')
input('')









###########################################
# Regression
###########################################
# Regression : target is elec_demand
# 'Elec_demand_cluster' is kept to build 3 regression functions
data = df[df.columns.difference(['location', 'elec_demand', 'elec_prediction','energ_indep'])]
target = df['elec_demand']

# Normalization occurs later in pipelines (StandardScaler(),...)
X_train, X_test, y_train, y_test = train_test_split( 
    data, target, test_size=0.3, shuffle=True, random_state=1)


# Storing MSE=-NMSE score for each model (optimized separately through a grid/random search)
regression_dict= dict()
models_simple_linear = regression_simple_linear(X_train, y_train)
models_regularized_polyfeat_ridge = regression_regularized_polyfeat_ridge(X_train, y_train)
regression_dict['Linear'] = (models_simple_linear[0], models_simple_linear[1], models_simple_linear[2], models_simple_linear[3])
regression_dict['Ridge'] = (models_regularized_polyfeat_ridge[0].best_estimator_, models_regularized_polyfeat_ridge[1].best_estimator_, models_regularized_polyfeat_ridge[2].best_estimator_, models_regularized_polyfeat_ridge[3])

fig, ax = plt.subplots()
plt.bar(x=list(regression_dict.keys()), height=[list(list(regression_dict.values())[0])[3],\
                                                list(list(regression_dict.values())[1])[3]],\
        color='g')
plt.xticks(rotation=45)
plt.xlabel('Regression models')
plt.ylabel('Mean squared error score (-)')
plt.title('Comparing regression models performance')
plt.show()


# Model chosen after min mean squared error score
model_regression_0 = min(regression_dict.values(), key=lambda sub: sub[3])[0]
model_regression_1 = min(regression_dict.values(), key=lambda sub: sub[3])[1]
model_regression_2 = min(regression_dict.values(), key=lambda sub: sub[3])[2]
best_score = min(regression_dict.values(), key=lambda sub: sub[3])[3]
model_name = [key for key, val in regression_dict.items() if val[3] == best_score][0]

print('\nRegression : model 0 characteristics :',  model_regression_0.get_params)
print('\nRegression : model 1 characteristics :',  model_regression_1.get_params)
print('\nRegression : model 2 characteristics :',  model_regression_2.get_params)
print('\nRegression : best overall score :',  best_score)

print('Confirm regression model choice ? (Y/N)')
answer = input('')
if(answer=='Y' or answer=='y') :
    pass
elif(answer=='N' or answer=='n'):
    if(model_name=='Linear'):
        model_regression_0 = regression_dict['Ridge'][0]
        model_regression_1 = regression_dict['Ridge'][1]
        model_regression_2 = regression_dict['Ridge'][2]
    elif(model_name=='Ridge'):
        model_regression_0 = regression_dict['Linear'][0]
        model_regression_1 = regression_dict['Linear'][1]
        model_regression_2 = regression_dict['Linear'][2]


ind_list_0 = df.loc[df['elec_demand_cluster'].isin([0])].index
df_tmp = df[df.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_0]
df['elec_prediction'].loc[ind_list_0] = model_regression_0.predict(df_tmp)

ind_list_1 = df.loc[df['elec_demand_cluster'].isin([1])].index
df_tmp = df[df.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_1]
df['elec_prediction'].loc[ind_list_1] = model_regression_1.predict(df_tmp)

ind_list_2 = df.loc[df['elec_demand_cluster'].isin([2])].index
df_tmp = df[df.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_2]
df['elec_prediction'].loc[ind_list_2] = model_regression_2.predict(df_tmp)


# y_pred = model_regression.predict(X_test) 
# print("\nRegression : RMSE on the test set: %0.3f" % np.sqrt(mean_squared_error(y_test, y_pred)))

print('\nRegression : end of section : press a key to continue')
input('')

##############
# Data viz
##############

plot_regression(model_classification, model_regression_0, model_regression_1, \
                model_regression_2, X_test, y_test)


###########################################
# Prediction : clustering, classification and regression for new raw data
###########################################
# New parameters are considered within the scope of application of original dataset
df_new = {
    'location' : encoder.transform(np.array(['east','centre','north'])),
    'aver_temp_fall_winter' : np.array([15,1,8]),
    'aver_temp_spring_summer' : np.array([27,22,16]),
    'energ_indep' : np.array([0.0,0.2,0.2]),
    'elec_demand' : np.array([0,0,0]),
    'elec_demand_cluster' : np.array([-1,-1,-1]), 
    'elec_prediction' : np.array([-1,-1,-1]), 
    }

df_new = pd.DataFrame.from_dict(df_new)

df_tmp = df_new[df_new.columns.difference(['location', 'elec_demand_cluster', 'elec_demand', 'elec_prediction'])]
df_tmp_1 = pd.DataFrame(scaler_data.transform(df_tmp), columns=df_tmp.columns)

df_tmp = pd.DataFrame(scaler_target.transform(df_new['elec_demand'].values.reshape(-1, 1)), columns=[df_new['elec_demand'].name]) 

df_tmp = pd.concat([df_tmp_1, df_tmp], join = 'outer', axis = 1)
df_tmp_1 = df_new[['location', 'elec_demand_cluster', 'elec_prediction']]
df_new = pd.concat([df_tmp, df_tmp_1], join = 'outer', axis = 1)


df_tmp = df_new[df_new.columns.difference(['location', 'elec_demand', 'elec_demand_cluster','elec_prediction','energ_indep'])]
# Classification : which label fits the best to these inputs ?
df_new['elec_demand_cluster'] = model_classification.predict(df_tmp) 


# Regression : which will be the elec_demand according to these inputs and the label ?
ind_list_00 = df_new.loc[df_new['elec_demand_cluster'].isin([0])].index
df_tmp = df_new[df_new.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_00]
df_new['elec_prediction'].loc[ind_list_00] = model_regression_0.predict(df_tmp)

ind_list_11 = df_new.loc[df_new['elec_demand_cluster'].isin([1])].index
df_tmp = df_new[df_new.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_11]
df_new['elec_prediction'].loc[ind_list_11] = model_regression_1.predict(df_tmp)

ind_list_22 = df_new.loc[df_new['elec_demand_cluster'].isin([2])].index
df_tmp = df_new[df_new.columns.difference(['location', 'elec_demand','elec_demand_cluster','elec_prediction','energ_indep'])]
df_tmp = df_tmp.loc[ind_list_22]
df_new['elec_prediction'].loc[ind_list_22] = model_regression_2.predict(df_tmp)


    
###########################################
# Electricity supply forecast
###########################################

# Inverse transform

df_tmp = df[df.columns.difference(['location', 'elec_demand_cluster', 'elec_demand', 'elec_prediction'])]
df_tmp_1 = pd.DataFrame(scaler_data.inverse_transform(df_tmp), columns=df_tmp.columns)

df_tmp = pd.DataFrame(scaler_target.inverse_transform(df['elec_demand'].values.reshape(-1, 1)), columns=[df['elec_demand'].name]) 
df_tmp_2 = pd.concat([df_tmp_1, df_tmp], join = 'outer', axis = 1)

df_tmp = pd.DataFrame(scaler_target.inverse_transform(df['elec_prediction'].values.reshape(-1, 1)), columns=[df['elec_prediction'].name]) 
df_tmp_1 = pd.concat([df_tmp_2, df_tmp], join = 'outer', axis = 1)

df_tmp = df[['location', 'elec_demand_cluster']]
df = pd.concat([df_tmp, df_tmp_1], join = 'outer', axis = 1)




df_tmp = df_new[df.columns.difference(['location', 'elec_demand_cluster', 'elec_demand', 'elec_prediction'])]
df_tmp_1 = pd.DataFrame(scaler_data.inverse_transform(df_tmp), columns=df_tmp.columns)

df_tmp = pd.DataFrame(scaler_target.inverse_transform(df_new['elec_demand'].values.reshape(-1, 1)), columns=[df_new['elec_demand'].name]) 
df_tmp_2 = pd.concat([df_tmp_1, df_tmp], join = 'outer', axis = 1)

df_tmp = pd.DataFrame(scaler_target.inverse_transform(df_new['elec_prediction'].values.reshape(-1, 1)), columns=[df_new['elec_prediction'].name]) 
df_tmp_1 = pd.concat([df_tmp_2, df_tmp], join = 'outer', axis = 1)

df_tmp = df_new[['location', 'elec_demand_cluster', 'elec_prediction']]
df_new = pd.concat([df_tmp, df_tmp_1], join = 'outer', axis = 1)


# Dict with (cluster_population, average_cluster_predicted_elec_demand)
total_demand_predict_dict = dict()

total_demand_predict_dict['cluster_0'] = [df['elec_demand_cluster'].value_counts()[0] + \
           df_new['elec_demand_cluster'].value_counts()[0],\
               np.mean(df['elec_prediction'].loc[ind_list_0].sum()+\
                       df_new['elec_prediction'].loc[ind_list_00].sum())]

total_demand_predict_dict['cluster_1'] = [df['elec_demand_cluster'].value_counts()[1]+\
           df_new['elec_demand_cluster'].value_counts()[1],\
               np.mean(df['elec_prediction'].loc[ind_list_1].sum()+\
                       df_new['elec_prediction'].loc[ind_list_11].sum())]

total_demand_predict_dict['cluster_2'] = [df['elec_demand_cluster'].value_counts()[2]+\
           df_new['elec_demand_cluster'].value_counts()[2],\
               np.mean(df['elec_prediction'].loc[ind_list_2].sum()+\
                       df_new['elec_prediction'].loc[ind_list_22].sum())]


total_elec_demand = \
    total_demand_predict_dict['cluster_0'][0] * total_demand_predict_dict['cluster_0'][1] + \
        total_demand_predict_dict['cluster_1'][0] * total_demand_predict_dict['cluster_1'][1] + \
            total_demand_predict_dict['cluster_2'][0] * total_demand_predict_dict['cluster_2'][1] 
        

print('\nTotal electricity predicted demand (kwh) : ', total_elec_demand)
    
################
# Data viz
################
fig, ax = plt.subplots()
plt.scatter(df['aver_temp_fall_winter'], df['elec_prediction'], marker='o', c='black')
plt.scatter(df['aver_temp_fall_winter'], df['elec_demand'], marker='o', c='blue')
plt.xlabel('Average temperature in fall-winter (°C)')
plt.ylabel('Electricity demand (kWh)')
plt.legend(['Predicted elec demand','Real elec demand'])
plt.title('Regression')
plt.show()
 
difference = df['elec_demand'] - df['elec_prediction'] 
fig, ax = plt.subplots()
plt.scatter(df['aver_temp_fall_winter'], difference, c='red')
plt.xlabel('Average temperature in fall-winter (°C)')
plt.ylabel('Real demand - predicted elec consumption (kWh)')
plt.title('Assessing error between real data and prediction')
plt.show()

fig, ax = plt.subplots()
plt.scatter(df['aver_temp_spring_summer'], difference, c='red')
plt.xlabel('Average temperature in spring-summer (°C)')
plt.ylabel('Real demand - predicted elec consumption (kWh)')
plt.title('Assessing error between real data and prediction')
plt.show()
  
df['location'] = encoder.inverse_transform(df['location'])
df_new['location'] = encoder.inverse_transform(df_new['location'])