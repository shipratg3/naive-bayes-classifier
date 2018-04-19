# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:29:19 2018

@author: shipra tyagi
"""

def splitDataset(dataset, splitRatio):
    from sklearn.model_selection import train_test_split
    X_test, X_train = train_test_split(dataset, test_size = splitRatio, random_state = 5)
    return(X_train,X_test)
#%%
def drop_columns(dataset):
    import pandas as pd
    cname=list(dataset)
    for i in cname:
        if(len(pd.unique(dataset[i]))==len(dataset[i])):
            data=dataset.drop(i,axis=1)
        else:
            data=dataset
    return data
#%%
def search_prob(t,Class,trainingSet):
    import numpy as np
    h=np.unique(trainingSet[Class])
    prob=[]
    for i in h:
        r=0
        for j in list(trainingSet):
            d=frequency_tables(Class,trainingSet)
            #print(d)
            w=0
            for k in np.unique(trainingSet[j]):
                if(t==k):
                    c=d[r]
                    p=c[i][w]
                w+=1
            r+=1
        prob.append(p)
    return prob
#%%
def prediction_function(trainingSet,testSet,Class):
    import numpy as np
    import pandas as pd
    freq_tb=frequency_tables(Class,trainingSet)
    test=testSet.drop(Class,axis=1)
    l=[]
    for i in range(len(np.unique(trainingSet[Class]))):
        l.append(freq_tb[-1]["All"][i])
    test_data=[list(i) for i in test.values] #list of test values
    resy=[]
    for i in test_data:
        y=[]
        for j in i:
            no_yes=search_prob(j,Class,trainingSet)
            y.append(no_yes)
        dk=pd.DataFrame(y)
        mn=[]
        for i in list(dk):
            mn.append(np.product(np.array(dk[i])))    
        resy.append(mn)
    ry=[]
    for i in resy:
        a=0
        rn=[]
        for j in i:
            rn.append(j*l[a])
            a+=1
        ry.append(rn)
    dm=pd.DataFrame(ry)
    kmr=[max(list(i)) for i in dm.values]
    kmn=[list(j) for j in dm.values]
    tm=[]
    for i in range(len(kmr)):
        tm.append(kmn[i].index(kmr[i]))
    res=pd.unique(trainingSet[Class])
    pred=[]
    for i in tm:
        pred.append(res[i])
    return pred  
#%%  
def error(existr,pred):
    import numpy as np
    return pred==np.array(existr)
#%%
def normalize_data(df):
    import numpy as np
    df=df.dropna(axis=0)
    cname=list(df)
    for i in cname:
        if df[i].dtype==float:
            for j in range(len(df[i])):
                if df[i][j]<=np.sum(df[i])/4:
                    df[i][j]=0
                elif df[i][j]>3*(np.sum(df[i]))/4:
                    df[i][j]=3
                elif df[i][j]>(np.sum(df[i])/4) and df[i][j]<=np.sum(df[i])/2:
                    df[i][j]=1
                elif df[i][j]<=3*(np.sum(df[i])/4) and df[i][j]>np.sum(df[i])/2:
                    df[i][j]=2 
    for i in cname:
        if df[i].dtype==float:
            df[i][df[i]==0.0]='a'
            df[i][df[i]==1.0]='b'
            df[i][df[i]==2.0]='c'
            df[i][df[i]==3.0]='d'
    return df
#%%
def frequency_tables(Class,trainingSet):
    b=[]
    import pandas as pd
    c=list(trainingSet)
    for j in c:
        if j!=Class:
            s=pd.crosstab(index=trainingSet[j],columns=[trainingSet[Class]],margins=True)
            a=list(s)
            for i in a:
                s[i]=s[i]/s[i][len(pd.unique(trainingSet[j]))]
            #b.append(s)
        elif j==Class:
            s=pd.crosstab(index=trainingSet[Class],columns=[trainingSet[Class]],margins=True)
            s=s/s["All"][len(pd.unique(trainingSet[Class]))]
        b.append(s)
            
    return b
#%%
def naive_bayes(df,splitPercentage,Class):
    splitRatio=splitPercentage/100
    dataset=drop_columns(df)
    dataset=normalize_data(df)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    existr=testSet[Class]
    pred=prediction_function(trainingSet,testSet,Class)
    print('predictions are :',pred)
    err=error(existr,pred)
    print('Match with prediction :',err)