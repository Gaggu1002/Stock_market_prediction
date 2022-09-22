import pandas_datareader.data as web
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings 

warnings.filterwarnings("ignore")
sca = MinMaxScaler(feature_range=(0,1))


def stock_res(name):
    
    data = web.DataReader(  name+".NS" , "yahoo")
    data =data["Close"]
    data = np.array(data).reshape(-1,1)
    test_size = 0.40
    train_size = int(len(data) * (1- test_size))
    x_train_data,x_test_data = data[0:train_size],data[train_size:len(data)]
    def create_dataset(data , timeSkip = 100):
        data_x=[]
        data_y = []
        for i in range (len(data) - timeSkip):
            a = data[i :(i+timeSkip), 0]
            data_x.append(a)
            data_y.append(data[i+timeSkip,0])
        return np.array(data_x) , np.array(data_y)
    x_train,y_train = create_dataset(x_train_data,150)
    x_test,y_test = create_dataset(x_test_data,150)
    model = []
    model.append(("linear Regressior",LinearRegression()))
    model.append(("lasso Regressior",Lasso(alpha=2)))
    model.append(("Ridge CV", RidgeCV()))
    model.append(("Decision Tree", DecisionTreeRegressor()))
    model.append(("Random Forest", RandomForestRegressor()))
    model.append(("Extree",ExtraTreesRegressor()))
    model.append(("ADA BOOST",AdaBoostRegressor()))
    model.append(("Gradiet Boosting",GradientBoostingRegressor()))
    model.append(("K Neigh ",KNeighborsRegressor()))
    model.append(("SVM",SVR()))
    model.append(("MLP",MLPRegressor()))
    ques_data = web.DataReader(name+".NS" , "yahoo")
    ques = ques_data.Close[-150:,]
    ques = np.array(ques).reshape(-1,150)
    model_name = []
    model_error = []
    res = []

    for q,l in model:
        l.fit(x_train,y_train)
        model_name.append(q)
        res.append(l.predict(ques))
        model_error.append(np.sqrt(mean_squared_error(l.predict(x_test),y_test)))
    return model_name ,  np.round( model_error,2) , np.round( np.array(res).tolist() , 2)




def checker(stock):
    try:
        data = web.DataReader(stock +".NS" , "yahoo")
        a = "1"
        last_name = np.round( data["Close"][-1] , 4)
    except :
        a = "0"
        last_name = "0"
    return a , last_name 


lst = pd.read_csv("list for stock.csv")
def lister():
    return lst["Security Id"],lst["Issuer Name"] ,lst["Industry"] 





