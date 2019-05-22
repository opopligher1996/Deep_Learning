import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from django.http import JsonResponse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from sklearn.externals import joblib
from minibus_ml.eta import NeuralNetModel
from django.http import HttpResponse

def new_neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,200]))
    b_1 = tf.Variable(tf.zeros([200]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    #layer_1 = tf.nn.tanh(layer_1)
    layer_1 = tf.nn.relu(layer_1)

    W_2 = tf.Variable(tf.random_uniform([200,20]))
    b_2 = tf.Variable(tf.zeros([20]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    #layer_2 = tf.nn.tanh(layer_2)
    layer_2 = tf.nn.relu(layer_2)

    W_O = tf.Variable(tf.random_uniform([20,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_2,W_O), b_O)
    return output,W_O

def predictResult(input,route,seq,predict):
    print('predictResult')
    df = pd.read_csv('minibus_ml/data_'+route+seq+'_'+predict+'.csv')
    df = df[(df['time_different'] > 0)]
    print(df)
    xs = tf.placeholder("float")
    output,W_O = new_neural_net_model(xs,4)
    scaler = MinMaxScaler()
    saver = tf.train.Saver()
    sesson = tf.Session()
    saver.restore(sesson, "/tmp/model"+route+seq+predict+"/model"+route+seq+predict+".ckpt")
    sesson.run(tf.global_variables())
    temp = pd.DataFrame([input],columns = ["distance", "WeekDay", "Hour", "v"])
    b = scaler.fit_transform(df.drop(['time_different'],axis=1).as_matrix())
    df_temp = scaler.transform(temp.as_matrix())
    input_pre = sesson.run(output, feed_dict={xs:df_temp})
    c = scaler.fit_transform(df['time_different'].as_matrix().reshape(-1, 1))
    result = scaler.inverse_transform(input_pre)
    return result

def predict(request):
    #http://127.0.0.1:8000/cal/?route=11M&seq=1&distance=3.0888946777375796&weekDay=2&hour=20&v=35.08351036981856&predict=4
    print('request')
    print(request)
    route = request.GET.get('route')
    seq = request.GET.get('seq')
    distance = request.GET.get('distance')
    hour = request.GET.get('hour')
    v = request.GET.get('v')
    predict = request.GET.get('predict')
    weekDay = request.GET.get('weekDay')
    t_data = [distance,weekDay,hour,v]
    # t_data = [3.0888946777375796,2,21,35.08351036981856]
    sesson = tf.Session()
    result = predictResult(t_data,route,seq,predict)
    print('result')
    print(result[0][0])
    resp = {}
    #'status': 200, 'response':result[0]
    resp['status'] = 200
    resp['response'] = result[0][0]
    return JsonResponse({'status':200, 'response':str(result[0][0])})
