#Import Library
from sklearn import svm
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(dataset_path,label):
    train=pd.read_excel(dataset_path)
    # train.dropna(inplace=True)
    y = train[label]
    X = train[['number_of_characters', 'is_entity_critical_word', 'number_of_dominated_nodes',
               'complexity_score', 'max_dependency_distance', 'number_of_senses_in_wordnet']]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.1, random_state=420)
    model = svm.SVC(kernel='linear', C=1, gamma=1)
    print(Xtrain,Ytrain)
    model.fit(Xtrain, Ytrain.astype('str'))
    print(model.score(Xtrain, Ytrain))

    #Predict Output
    y_pred= model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(Ytest, y_pred))
    print('RMSE:', rmse)
    return rmse

def svm_2(dataset_path,label):
    # 从sklearn.svm中导入支持向量机（回归）模型。
    from sklearn.svm import SVR
    train = pd.read_excel(dataset_path)
    # train.dropna(inplace=True)
    y = train[label]
    X = train[['number_of_characters', 'is_entity_critical_word', 'number_of_dominated_nodes',
               'complexity_score', 'max_dependency_distance', 'number_of_senses_in_wordnet']]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.1, random_state=420)
    # 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
    linear_svr = SVR(kernel='linear')
    linear_svr.fit(Xtrain, Ytrain)
    linear_svr_y_predict = linear_svr.predict(Xtest)

    # 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
    poly_svr = SVR(kernel='poly')
    poly_svr.fit(Xtrain, Ytrain)
    poly_svr_y_predict = poly_svr.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(Ytest, poly_svr_y_predict))
    print('RMSE:', rmse)
    # 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
    # rbf_svr = SVR(kernel='rbf')
    # rbf_svr.fit(Xtrain, Ytrain)
    # rbf_svr_y_predict = rbf_svr.predict(Xtest)

label='avg_word_first_duration'
print('=====avg_word_first_duration=======')
print('Provo')
# get_rmse('Gaze_data/Provo_avg.xlsx',label)
svm_2('Gaze_data/Provo_avg.xlsx',label)
svm_2('Gaze_data/GECO_avg.xlsx',label)
svm_2('Gaze_data/EA_avg.xlsx',label)
print('======avg_word_total_reading_time======')
label='avg_word_total_reading_time'
svm_2('Gaze_data/Provo_avg.xlsx',label)
svm_2('Gaze_data/GECO_avg.xlsx',label)
svm_2('Gaze_data/EA_avg.xlsx',label)
print('======avg_word_go_past_time======')
label='avg_word_go_past_time'
svm_2('Gaze_data/Provo_avg.xlsx',label)
svm_2('Gaze_data/GECO_avg.xlsx',label)
svm_2('Gaze_data/EA_avg.xlsx',label)