# Importing the libraries
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import DataConversionWarning
import pickle
warnings.filterwarnings(action='ignore',category=DataConversionWarning)


df=pd.read_csv('data.csv',index_col=0)
cols=list(df.columns)[53:87]+['Skill Moves']
attributes=df[cols]
attributes.head()
workrate=df['Work Rate'].str.get_dummies(sep='/ ')
attributes=pd.concat([attributes,workrate],axis=1)
#print(attributes.shape)
attributes=attributes.dropna()
player_info=attributes.copy()
player_info['Name']=df['Name']

scaler=StandardScaler()
X=scaler.fit_transform(attributes)

recommendations=NearestNeighbors(n_neighbors=6,algorithm='ball_tree').fit(X)

player_indices=recommendations.kneighbors(X)[1]

def get_index(x):
    return (player_info[player_info['Name'].str.contains(x)].index.tolist()[0],player_info[player_info['Name'].str.contains(x)]['Name'].tolist()[0])
def recommend_me(player):
    ind,name=get_index(player)

    out=[]
    #print('Here are 5 players similar to ',name, '\n')
    
    for i in player_indices[ind][1:]:
        out.append(player_info.iloc[i]['Name'])

    return out
print(recommend_me('Chhet'))