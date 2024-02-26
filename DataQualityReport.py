#%% IMPORT NECESSARY LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%% IMPORT THE DATASET
ruta = 'C:/Users/jsramirez/datsets_pm/heart+disease'
col = ['age', 'sex', 'cp','trestbps', 'chol', 'fbs',
           'restecg', 'thalach', 'exang', 'oldpeak','slope',
           'ca', 'thal', 'num']
data = pd.read_csv(ruta+'/processed.cleveland.data', header=None,na_values='?')
data.columns = col

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
# %% Replacement of missing values

moda = data.mode().iloc[0]
data_filled = data.fillna(moda)
report = DQR(data_filled)
report

# %% Logistic regression

X, y = data_filled.iloc[:,:-1], data_filled.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

Logreg = LogisticRegression()

Logreg.fit(X_train,y_train)

y_pred_test = Logreg.predict(X_test)
y_pred_train = Logreg.predict(X_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'El accuracy de la regresión logística en el conjunto de prueba es: {accuracy_test}')
print(f'El accuracy de la regresión logística en el conjunto de entrenamiento es: {accuracy_train}')

#%% dimensionality reduction by PCA
pca=PCA().fit(X)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Numero de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.show()
#%% Logistic Regression with PCA
ACP = PCA(n_components=2)
X_pca = pd.DataFrame(ACP.fit_transform(X), columns=['C1', 'C2'])
Xpca_train, Xpca_test, y_train, y_test = train_test_split(X_pca,y, test_size=0.2, random_state=42)

Logreg = LogisticRegression()

Logreg.fit(Xpca_train,y_train)

y_pred_test = Logreg.predict(Xpca_test)
y_pred_train = Logreg.predict(Xpca_train)
accuracy_test_pca = accuracy_score(y_test, y_pred_test)
accuracy_train_pca = accuracy_score(y_train, y_pred_train)
print(f'El accuracy de la regresión logística en el conjunto de prueba es: {accuracy_test_pca}')
print(f'El accuracy de la regresión logística en el conjunto de entrenamiento es: {accuracy_train_pca}')


# %% LDA

lda = LDA(store_covariance=True,n_components=None)
lda = lda.fit(X,y)
plt.plot(np.cumsum(lda.explained_variance_ratio_))
plt.xlabel('Numero de componentes')
plt.ylabel('Varianza explicada acumulada')
plt.show()
#%% Logistic Regression with LDA

X_lda = lda.transform(X)
Xlda_train, Xlda_test, y_train, y_test = train_test_split(X_pca,y, test_size=0.2, random_state=42)

Logreg = LogisticRegression()

Logreg.fit(Xlda_train,y_train)

ylda_pred_test = Logreg.predict(Xlda_test)
ylda_pred_train = Logreg.predict(Xlda_train)
accuracy_test_lda = accuracy_score(y_test, ylda_pred_test)
accuracy_train_lda = accuracy_score(y_train, ylda_pred_train)
print(f'El accuracy de la regresión logística en el conjunto de prueba es: {accuracy_test_lda}')
print(f'El accuracy de la regresión logística en el conjunto de entrenamiento es: {accuracy_train_lda}')

# %%
