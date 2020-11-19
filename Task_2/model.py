
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.models import Model
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def read_data(path):
    propulsion_data=pd.read_csv(path)
    return propulsion_data

def remove_data(data):
    data=data.drop(columns=["Unnamed: 0"])
    return data

def seperate_data(data):
    data=data.values
    X = data[:,0:16]
    Y = data[:,16]
    Y1 = data[:,17]
    return X,Y,Y1

def split_data(X,Y,Y1):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)
    X_Train1, X_Test1, Y_Train1, Y_Test1 = train_test_split(X, Y1, test_size=0.3)
    return X_Train,X_Test,Y_Train,Y_Test,X_Train1,X_Test1,Y_Train1,Y_Test1

def model():
    #X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)
    #X_Train1, X_Test1, Y_Train1, Y_Test1 = train_test_split(X, Y1, test_size=0.3)

    input_0 = Input(shape=(16,))
    input_1 = Input(shape=(16,))
    #merged = keras.layers.Concatenate(axis=1)([input1, input2])
    dense_0 = Dense(6,init='uniform', activation='relu')(input_0)
    dense_1 = Dense(6,init='uniform', activation='relu')(input_1)
    dropout_0=Dropout(0.2)(dense_0)
    dropout_1=Dropout(0.2)(dense_1)
    dense_0 = Dense(4,init='uniform', activation='relu',kernel_constraint=maxnorm(3))(dropout_0)
    dense_1 = Dense(4,init='uniform', activation='relu',kernel_constraint=maxnorm(3))(dropout_1)
    dropout_0=Dropout(0.2)(dense_0)
    dropout_1=Dropout(0.2)(dense_1)
    dense_0 = Dense(2,init='uniform', activation='relu')(dropout_0)
    dense_1 = Dense(2,init='uniform', activation='relu')(dropout_1)
    dense_0 = Dense(1,init='uniform', activation='relu')(dense_0)
    dense_1 = Dense(1,init='uniform', activation='relu')(dense_1)
    #output=   Concatenate(axis=1)([dense_0,dense_1])
    model=    Model(inputs=[input_0,input_1],outputs=[dense_0,dense_1])
    model.compile(loss=root_mean_squared_error, optimizer='adam')
    return model

if __name__=="__main__":

    path=input('Enter the path of the dataset')
    propulsion_data=read_data(path)
    propulsion_data=remove_data(propulsion_data)
    X,Y,Y1=seperate_data(propulsion_data)
    X_Train,X_Test,Y_Train,Y_Test,X_Train1,X_Test1,Y_Train1,Y_Test1=split_data(X,Y,Y1)
    model=model()
    model.fit([X_Train,X_Train1],[Y_Train,Y_Train1], epochs=10, batch_size=10)
    score=model.evaluate([X_Test,X_Test1],[Y_Test,Y_Test1])
    print('score-->',np.array([100,100,100])-np.array(score))






