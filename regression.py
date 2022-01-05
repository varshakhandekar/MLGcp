import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_squared_error
import pickle
boston = load_boston()
bos = pd.DataFrame(boston.data,columns=boston.feature_names)
bos.isnull().sum()
bos['Price']=boston.target
fig,axs=plt.subplots(1,2,sharey=True)
bos.plot(kind='scatter',x='Price',y='RM',ax=axs[0],figsize=(10,6))
bos.plot(kind='scatter',x='Price',y='LSTAT',ax=axs[1])
#bos.plot(kind='scatter',x='TV',y='newspaper',ax=axs[2],figsize=(10,6))
fig=plt.figure(figsize=(20,30))
correlation_matrix = bos.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
sns.pairplot(bos, hue = 'Price')
# displaying the plot
plt.show()
X=bos[['RM','LSTAT','PTRATIO','INDUS']]
y=bos['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.30,random_state=323)

lm=LinearRegression()
lm.fit(xtrain,ytrain)
filename='Regression_Assignment_model.pickle'
pickle.dump(lm,open(filename,'wb'))

train_predictedy=lm.predict(xtrain)

train_r_square=r2_score(ytrain,train_predictedy)

print('Model Performance for Training Data')
print('-'*60)
#print('Training data root mean square error is',train_rmse)
print('Training datar-square is',train_r_square)

test_predictedy=lm.predict(xtest)

test_rmse=np.sqrt(mean_squared_error(ytest,test_predictedy))
test_r_square=r2_score(ytest,test_predictedy)

print('Model Performance for Test Data')
print('-'*60)
print('Test data root mean square error is',test_rmse)
print('Test data r-square is',test_r_square)

lassocv=LassoCV(alphas=None,cv=10,max_iter=10000,normalize=True)
lassocv.fit(xtrain,ytrain)
alpha=lassocv.alpha_

lasso_reg=Lasso(alpha)
lasso_reg.fit(xtrain,ytrain)

lasso_reg.score(xtrain,ytrain)
lasso_reg.score(xtest,ytest)


