
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error


def read_data(path):
    car_data=pd.read_csv(path)
    return car_data

def fill_na(car_data):
    car_data['volume(cm3)'].fillna(2103.201676257193,inplace=True)
    z=car_data['segment'].mode()
    car_data['segment'].fillna(z[0],inplace=True)
    z=car_data['drive_unit'].mode()[0]
    car_data['drive_unit'].fillna(z,inplace=True)
    return car_data

def convt_data(car_data):
    car_data_cat=pd.DataFrame()
    for i in car_data.select_dtypes(include='object').columns:
        label_encode=LabelEncoder()
        car_data_cat[i]=label_encode.fit_transform(car_data[i])

    car_data_cont=car_data.select_dtypes(exclude='object').drop(columns=['Unnamed: 0'])
    car_data_=pd.concat([car_data_cont.drop(columns=['year']),car_data_cat],axis=1)
    X=car_data_.drop(columns=['priceUSD'])
    y=car_data_['priceUSD']
    return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
    return X_train,X_test,y_train,y_test

def train_predict(X_train,y_train,X_test):
    rand_forest=RandomForestRegressor()
    rand_forest.fit(X_train,y_train)
    pred=rand_forest.predict(X_test)
    return pred


if __name__=="__main__":

    path=input('Enter the path of the dataset')
    car_data=read_data(path)
    car_data=fill_na(car_data)
    X,y=convt_data(car_data)
    X_train,X_test,y_train,y_test=split_data(X,y)
    prediction=train_predict(X_train,y_train,X_test)
    print(mean_absolute_error(y_test,prediction),mean_squared_error(y_test,prediction))



