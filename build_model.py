# Importing the libraries
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import DataConversionWarning
import pickle
from numpy import savetxt
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

savetxt('out.csv',player_indices,delimiter=',')

