import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("epldata_final.csv")
df2 = df.copy()

corrmatrix = df2.corr()  #Only those columns that have good correlation are selected.
#print(corrmatrix)

#decide Dependent and Independent Variable.
X = df2.loc[:,["market_value","page_views","fpl_points"]]
X = X.iloc[:,:].values
y = df2.iloc[:,7].values

#You can use divide the X into Xtrain and Xtest but I opted to usethe entire dataset.
'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
'''

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)  #fit the regressor

# Predicting the Test set results.
y_pred = regressor.predict(X)

v = list(range(0,461))  #just took the length of the entire number of data.
plt.scatter(v,y,label = 'Given Value')
plt.scatter(v,y_pred,c = 'yellow',label = 'Predicted Values')
plt.ylabel("FPL Value")
plt.legend()
plt.show()




