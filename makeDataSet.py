# %pylab inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from statsmodels.stats.outliers_influence import variance_inflation_factor


def make_dataset():

    np.random.seed(1)

    location = []
    for name in ['north', 'south', 'west', 'east', 'centre'] : 
        location+=[name for i in range(100)]
    location = np.array(location).reshape((500,1))
    
    # aver_temp_fall_winter =    np.random.uniform(0,10,size=500).reshape((500,1))
    # aver_temp_spring_summer =  np.random.uniform(10,30,size=500).reshape((500,1))
    # energ_indep =              np.random.uniform(0,0.2,size=500).reshape((500,1))
    
    # array = np.concatenate(((location, aver_temp_fall_winter)), axis=1)
    # array = np.concatenate(((array, aver_temp_spring_summer)), axis=1)
    # array = np.concatenate(((array, energ_indep)),axis=1)

    aver_temp_fall_winter_N =    (np.random.uniform(5,8,size=100)+2*np.random.rand()).reshape((100,1))
    aver_temp_fall_winter_S =    (np.random.uniform(12,15,size=100)+3*np.random.rand()).reshape((100,1))
    aver_temp_fall_winter_W =    (np.random.uniform(5,8,size=100)+2*np.random.rand()).reshape((100,1))
    aver_temp_fall_winter_E =    (np.random.uniform(0,3,size=100)+np.random.rand()).reshape((100,1))
    aver_temp_fall_winter_C =    (np.random.uniform(0,3,size=100)+np.random.rand()).reshape((100,1))
    
    aver_temp_spring_summer_N =  (np.random.uniform(12,18,size=100)+2*np.random.rand()).reshape((100,1))
    aver_temp_spring_summer_S =  (np.random.uniform(22,25,size=100)+3*np.random.rand()).reshape((100,1))
    aver_temp_spring_summer_W =  (np.random.uniform(12,18,size=100)+2*np.random.rand()).reshape((100,1))
    aver_temp_spring_summer_E =  (np.random.uniform(19,23,size=100)+np.random.rand()).reshape((100,1))
    aver_temp_spring_summer_C =  (np.random.uniform(19,23,size=100)+np.random.rand()).reshape((100,1))
    
    energ_indep =              np.random.uniform(0,0.2,size=500).reshape((500,1))
    

    aver_temp_fall_winter = np.concatenate(((aver_temp_fall_winter_N, aver_temp_fall_winter_S)), axis=0)
    aver_temp_fall_winter = np.concatenate(((aver_temp_fall_winter, aver_temp_fall_winter_W)), axis=0)
    aver_temp_fall_winter = np.concatenate(((aver_temp_fall_winter, aver_temp_fall_winter_E)), axis=0)
    aver_temp_fall_winter = np.concatenate(((aver_temp_fall_winter, aver_temp_fall_winter_C)), axis=0)
    
    aver_temp_spring_summer = np.concatenate(((aver_temp_spring_summer_N, aver_temp_spring_summer_S)), axis=0)
    aver_temp_spring_summer = np.concatenate(((aver_temp_spring_summer, aver_temp_spring_summer_W)), axis=0)
    aver_temp_spring_summer = np.concatenate(((aver_temp_spring_summer, aver_temp_spring_summer_E)), axis=0)
    aver_temp_spring_summer = np.concatenate(((aver_temp_spring_summer, aver_temp_spring_summer_C)), axis=0)
     
    array = np.concatenate(((location, aver_temp_fall_winter)), axis=1)
    array = np.concatenate(((array, aver_temp_spring_summer)), axis=1)
    array = np.concatenate(((array, energ_indep)), axis=1)

      
    # creating a list of column names
    column_values = ['location', 'aver_temp_fall_winter', 'aver_temp_spring_summer', 'energ_indep']
      
    # creating the dataframe
    df = pd.DataFrame(data = array,  
                      columns = column_values)
    
    df['location'] = df['location'].astype(str)
    df['aver_temp_fall_winter'] = df['aver_temp_fall_winter'].astype(float)
    df['aver_temp_spring_summer'] = df['aver_temp_spring_summer'].astype(float)
    df['energ_indep'] = df['energ_indep'].astype(float)
    


    
    df.loc[df['location']=='north', 'elec_demand'] = ( 
        2500*np.sqrt(df.loc[df['location']=='north', 'aver_temp_fall_winter']/5) + \
            2500*np.sqrt(df.loc[df['location']=='north', 'aver_temp_spring_summer']/20))* \
        (1 - df.loc[df['location']=='north', 'energ_indep'])
        
        
    df.loc[df['location']=='west', 'elec_demand'] = ( 
        2500*np.sqrt(df.loc[df['location']=='west', 'aver_temp_fall_winter']/5) + \
            2500*np.sqrt(df.loc[df['location']=='west', 'aver_temp_spring_summer']/20))* \
        (1 - df.loc[df['location']=='west', 'energ_indep'])
    
    
    
    
        
    df.loc[df['location']=='east', 'elec_demand'] = ( 
        2000*(df.loc[df['location']=='east', 'aver_temp_fall_winter']/5) + \
            2000*(df.loc[df['location']=='east', 'aver_temp_spring_summer']/20))* \
        (1 - df.loc[df['location']=='east', 'energ_indep'])
        
    df.loc[df['location']=='centre', 'elec_demand'] = ( 
        2000*(df.loc[df['location']=='centre', 'aver_temp_fall_winter']/5) + \
            2000*(df.loc[df['location']=='centre', 'aver_temp_spring_summer']/20))* \
        (1 - df.loc[df['location']=='centre', 'energ_indep'])
   
    
   
    
    df.loc[df['location']=='south', 'elec_demand'] = ( 
        800*(df.loc[df['location']=='south', 'aver_temp_fall_winter']/5)**2 + \
            800*(df.loc[df['location']=='south', 'aver_temp_spring_summer']/20)**2)* \
        (1 - df.loc[df['location']=='south', 'energ_indep'])
        
        
    
    df = df.sample(frac=1).reset_index(drop=True)

    return df


df = make_dataset()

df.to_csv('dataSetElec.csv',sep=',')


data = df.iloc[:,:-1]
target = df.iloc[:,-1]

var_0 = data['aver_temp_fall_winter'][:]
var_1 = data['aver_temp_spring_summer'][:]
fig, ax = plt.subplots()
scatter = ax.scatter(var_0, var_1, marker="o", c=target.iloc[:], 
            cmap=plt.cm.coolwarm, edgecolor="k")
# plt.xlim(min(var_0), max(var_0))
# plt.ylim(min(var_1), max(var_1))
plt.xlabel(var_0.name)
plt.ylabel(var_1.name)
plt.title("Data viz over two chosen features")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Electricity demand")
ax.add_artist(legend1)
plt.show()



var_0 = data['aver_temp_fall_winter'][:]
var_1 = data['energ_indep'][:]
fig, ax = plt.subplots()
scatter = ax.scatter(var_0, var_1, marker="o", c=target.iloc[:], 
            cmap=plt.cm.coolwarm, edgecolor="k")
# plt.xlim(min(var_0), max(var_0))
# plt.ylim(min(var_1), max(var_1))
plt.xlabel(var_0.name)
plt.ylabel(var_1.name)
plt.title("Data viz over two chosen features")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Electricity demand")
ax.add_artist(legend1)
plt.show()



fig, ax = plt.subplots()
ax.scatter(df['aver_temp_fall_winter'], df['elec_demand'], marker="o", color="k")
plt.xlabel(df['aver_temp_fall_winter'].name)
plt.ylabel(df['elec_demand'].name)
plt.title("Electricity demand over average temperature fall-winter")
plt.show()


# Step 1 multicollinearity : VIF score
# vif_df = pd.DataFrame(data={'Feature' : data.iloc[1:,:].columns, 
#             'VIF_score' : [variance_inflation_factor(data.iloc[1:,:].values, i) \
#                                       for i in range(data.iloc[1:,:].shape[1])]})  

# Step 2 multicollinearity : correlation matrix

# Pair plot
sns.pairplot(data)
plt.show()

# Correlation matrix
corr = data.corr()
f, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
plt.show()