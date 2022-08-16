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



def decision_tree(x_train, y_train):
    x_train = x_train * 10000
    y_train = y_train * 10000
    y_train = y_train.astype('int')
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    importance = classifier.feature_importances_
    indices = np.argsort(importance)[::-1] 
    df = pd.DataFrame(columns = ['ind variable', 'importance'])       
    df = df.append({"ind variable" : y_train.name, "importance" : " "}, ignore_index = True)
    
    for f in range(x_train.shape[1]):
        df = df.append({"ind variable" : x_train.columns[indices][f], "importance" : round(importance[indices[f]],6)}, ignore_index = True)
        
    return(df)     

def decision_tree_result(x_train, y_train, var, result):
    '''
    _trxain = x_train * 10000
    y_train = y_train * 10000
    y_train = y_train.astype('int')
    '''
    for j in range(len(var)):    
        df = pd.DataFrame(columns = ['ind variable', 'importance'])
        rfregressor = DecisionTreeRegressor(max_features = var[j])
        rfregressor.fit(x_train, y_train)
        importance = rfregressor.feature_importances_
        indices = np.argsort(importance)[::-1]
        df = df.append({"ind variable" : y_train.name, "importance" :  "var : " + str(var[j])}, ignore_index = True)
        for f in range(x_train.shape[1]):
            df = df.append({"ind variable" : x_train.columns[indices][f], "importance" : round(importance[indices[f]],6)}, ignore_index = True)
        result = result.append(df, ignore_index = True)
    return(result)

def decision_tree_result_df(x_train, y_train, result):

    df = pd.DataFrame(columns = ['ind variable', 'importance'])
    rfregressor = DecisionTreeRegressor()
    rfregressor.fit(x_train, y_train)
    importance = rfregressor.feature_importances_
    indices = np.argsort(importance)[::-1]
    df = df.append({"ind variable" : y_train.name, "importance" :  "var : " }, ignore_index = True)
    for f in range(x_train.shape[1]):
        df = df.append({"ind variable" : x_train.columns[indices][f], "importance" : round(importance[indices[f]],6)}, ignore_index = True)
    result = result.append(df, ignore_index = True)
    return(result)

def lasso(x_train, y_train, alpha, length) :
    lasso_coef = pd.DataFrame(columns= range(length))
    lasso_coef.columns = x_train.columns
    for a in alpha:
        lasso = Lasso(alpha = a, )
        lasso.fit(x_train, y_train)
        coef = lasso.coef_
        lasso_coef.loc[a] = coef
    return (lasso_coef)

def regression(x, y):
    Y = y
    X = x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    return(results)

def results_summary_to_dataframe(df,x, y):
    results = regression(x,y)
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
    rs.index = [y.name]                           
    results_df = pd.concat([rs, results_df[["coef","t", "p"]]])

    results_df = results_df.replace(np.nan, "", regex= True)
    results_df = round(results_df, 4)
    df = pd.concat([df, results_df])
    return(df)


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


def rfclass_result(x_train, y_train, est, var, result):   
    x_train = x_train * 10000
    y_train = y_train * 10000
    y_train = y_train.astype('int')
    for j in range(len(var)):    
        df = pd.DataFrame(columns = ['ind variable', 'importance'])
        for i in range(len(est)):
            rfregressor = RandomForestClassifier(n_estimators= est[i], max_depth = var[j])
            rfregressor.fit(x_train, y_train)
            importance = rfregressor.feature_importances_
            indices = np.argsort(importance)[::-1]
            df = df.append({"ind variable" : y_train.name, "importance" : "est : "+ str(est[i]) + ", var : " + str(var[j])}, ignore_index = True)
            for f in range(x_train.shape[1]):
                df = df.append({"ind variable" : x_train.columns[indices][f], "importance" : round(importance[indices[f]],6)}, ignore_index = True)
        result = result.append(df, ignore_index = True)
    return(result)   