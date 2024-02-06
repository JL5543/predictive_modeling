#%% IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#%% IMPORT THE DATASET
ruta = 'C:/Users/jsramirez/datsets_pm'
data = pd.read_csv(ruta+'/abalone.data', header=None)

with open(ruta+'/abalone.names','r') as file:
    columns = file.read()

print(columns)

names=['Sex','Lenght', 'Diameter', 'Height', 'Whole weight',
       'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

data.columns = names

#%% DATA QUALITY REPORT
def DQR(data):
    cols = pd.DataFrame(list(data.columns.values),
                        columns=['Names'],
                        index=list(data.columns.values))
    # List of data types
    dtyp = pd.DataFrame(data.dtypes,columns=['Type'])
    # Count of missing values per varible
    misval = pd.DataFrame(data.isnull().sum(),
                          columns=['Missing_values'])
    # Count of present data per variable
    presval = pd.DataFrame(data.count(),
                           columns=['Present_values'])
    # List of unique values
    unival = pd.DataFrame(columns=['Unique_value'])
    # List of min values
    minval = pd.DataFrame(columns=['Min_value'])
    # List of max values
    maxval = pd.DataFrame(columns=['Max_value'])
    for col in list(data.columns.values):
        unival.loc[col]=[data[col].nunique()]
        try:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        except:
            pass
    #Join the dataframes and return the result
    return cols.join(dtyp).join(misval).join(presval).join(unival).join(minval).join(maxval)

#%% OBTAINING THE DATA QUALITY REPORT
report = DQR(data)
report
# NO MISING VALUES

#%% ONE-HOT FOR VARIABLE SEX
sex_dummies = pd.get_dummies(data['Sex'], prefix='sex', drop_first=True)

data1 = pd.concat([sex_dummies, data], axis=1)

data1 = data1.drop('Sex', axis=1)
#%% SPLIT TRAINING AND TESTING DATASETS
X, y = data1.iloc[:,:-1], data1.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%% TRAIN THE LINEAR MODEL
model = LinearRegression()
model.fit(X_train, y_train)
y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
print(f'El RMSE para el conjunto de entrenamiento es de {rmse_train:.5f}')
print(f'El RMSE para el conjunto de prueba es de {rmse_test:.5f}')

#%% CORRELATION MATRIX
Corr_matrix = data1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(Corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.show()

#%% FEATURE SELECTION
#Eliminamos Lenght por diameter y shucked weight y viscera weight

data2 = data1.drop(['sex_M', 'Lenght', 'Shucked weight', 'Viscera weight'], axis=1)

X2, y2 = data2.iloc[:,:-1], data2.iloc[:,-1]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

#%% TRAIN THE LINEAR MODEL
model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_train_predict = model2.predict(X2_train)
y2_test_predict = model2.predict(X2_test)
rmse_train2 = np.sqrt(mean_squared_error(y2_train, y2_train_predict))
rmse_test2 = np.sqrt(mean_squared_error(y2_test, y2_test_predict))
print(f'El RMSE para el conjunto de entrenamiento es de {rmse_train2:.5f}')
print(f'El RMSE para el conjunto de prueba es de {rmse_test2:.5f}')


# %%
