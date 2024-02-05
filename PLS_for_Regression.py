#%% IMPORT NECESSARY LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

data = pd.concat([sex_dummies, data], axis=1)

data = data.drop('Sex', axis=1)
#%% SPLIT TRAINING AND TESTING DATASETS
X, y = data.iloc[:,:-1], data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%% TRAIN THE LINEAR MODEL
model = LinearRegression()
model.fit(X_train, y_train)
