from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import statsmodels.api as sm


def lasso(x_train, y_train, alpha) :
    lasso_coef = pd.DataFrame(columns= range(x_train.shape[1]))
    lasso_coef.columns = x_train.columns
    for a in alpha:
        lasso = Lasso(alpha = a, )
        lasso.fit(x_train, y_train)
        coef = lasso.coef_
        lasso_coef.loc[a] = coef
    return (lasso_coef)

def lasso_df(x_train, y_train, alpha) :
    lasso_coef = pd.DataFrame(columns= range(x_train.shape[1]))
    lasso_coef.columns = x_train.columns
    for i in range(y_train.shape[1]):    
        for a in alpha:
            lasso = Lasso(alpha = a, )
            lasso.fit(x_train, y_train.iloc[:,i])
            coef = lasso.coef_
            lasso_coef.loc[a] = coef
            
    return (lasso_coef)

def randomforest_result(x_train, y_train, est, var, result):   
    for j in range(len(var)):    
        df = pd.DataFrame(columns = ['ind variable', 'importance'])
        for i in range(len(est)):
            rfregressor = RandomForestRegressor(n_estimators= est[i], max_features = var[j])
            rfregressor.fit(x_train, y_train)
            importance = rfregressor.feature_importances_
            indices = np.argsort(importance)[::-1]
            df = df.append({"ind variable" : y_train.name, "importance" : "est : "+ str(est[i]) + ", var : " + str(var[j])}, ignore_index = True)
            for f in range(x_train.shape[1]):
                df = df.append({"ind variable" : x_train.columns[indices][f], "importance" : round(importance[indices[f]],6)}, ignore_index = True)
        result = result.append(df, ignore_index = True)

    return(result)

def results_summary_to_dataframe(df,x, y):
    for i in range(len(y.columns)):
        Y = y.iloc[:,i]
        X = x
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        pvals = results.pvalues
        coeff = results.params
        tvalues = results.tvalues
        rsquared = results.rsquared
        adjrsquared = results.rsquared_adj
        results_df = pd.DataFrame({"p":pvals,
                                    "t" : tvalues,   
                                     "coef":coeff,                                                   
                                    })
    
        rs = pd.DataFrame({"r2": rsquared, "adj-r2" : adjrsquared}, index = [0]) 
        rs.index = [Y.name]                           
        results_df = pd.concat([rs, results_df[["coef","t", "p"]]])
    
        results_df = results_df.replace(np.nan, "", regex= True)
        results_df = round(results_df, 4)
        df = pd.concat([df, results_df])
    return(df)
