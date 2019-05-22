import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.externals import joblib

#Neural Net Model
# 1. __init__() will get data_{route}M{seq}_{station_index}.csv to train the model
# 2. MinMaxScaler will be used to normalize the data
# 3.

class NeuralNetModel():

    route = ''

    seq = ''

    station = ''

    features = []

    predict = ''

    learning_rate = 0

    time = 0

    def __init__(self,predict, learning_rate, time, seq, route):
        print('start __init__')
        self.seq = seq
        self.route = route
        self.learning_rate = learning_rate
        self.time = time
        self.predict = predict
        self.sesson = tf.Session()
        self.graph = tf.Graph()
        df = pd.read_csv('data_'+route+seq+'_'+predict+'.csv')
        self.df = df[(df['time_different'] > 0)]
        print('self.df.shape[0] = ')
        event_count = self.df.shape[0]
        train_data_count = int(event_count*0.7)
        self.df_train = self.df[:train_data_count]
        self.df_test = self.df[train_data_count:]
        self.scaler = MinMaxScaler()
        b = self.scaler.fit_transform(self.df.drop(['time_different'],axis=1).as_matrix())
        self.X_train = self.scaler.transform(self.df_train.drop(['time_different'],axis=1).as_matrix())
        self.X_test = self.scaler.transform(self.df_test.drop(['time_different'],axis=1).as_matrix())

        c = self.scaler.fit_transform(self.df['time_different'].as_matrix().reshape(-1, 1))
        self.y_train = self.scaler.transform(self.df_train['time_different'].as_matrix().reshape(-1, 1))
        # self.X_test = self.scaler.transform(self.df_test.drop(['s3'],axis=1).as_matrix())
        self.y_test = self.scaler.transform(self.df_test['time_different'].as_matrix().reshape(-1, 1))

        self.xs = tf.placeholder("float")
        self.ys = tf.placeholder("float")

        print('end __init__')

    def train(self):
        self.output,self.W_O = NeuralNetModel.neural_net_model(self.xs,4)
        self.cost = tf.reduce_mean(tf.square(self.output-self.ys))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.correct_pred = tf.argmax(self.output, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))
        c_t = []
        c_test = []
        self.sesson.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.y_t = NeuralNetModel.denormalize(self.df,self.y_train,'time_different')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        plt.xlabel('Records')
        plt.ylabel('Time')
        plt.title('Minibus Eta prediction')
        self.ax.plot(range(len(self.y_train)), self.y_t,label='Original')
        plt.ion()
        for i in range(self.time):
            for j in range(self.X_train.shape[0]):
                self.sesson.run([self.cost,self.train],feed_dict={self.xs:self.X_train[j,:].reshape(1,4), self.ys:self.y_train[j]})

            try:
                self.ax.lines.remove(lines[0])
            except Exception:
                pass
            self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_train})
            self.pred = NeuralNetModel.denormalize(self.df,self.pred,'time_different')
            lines = self.ax.plot(range(len(self.y_train)), self.pred,'r-',label='Prediction')
            plt.legend(loc='best')
            plt.pause(0.1)

            c_t.append(self.sesson.run(self.cost, feed_dict={self.xs:self.X_train,self.ys:self.y_train}))
            c_test.append(self.sesson.run(self.cost, feed_dict={self.xs:self.X_test,self.ys:self.y_test}))
        save_path = self.saver.save(self.sesson, "/tmp/model"+self.route+self.seq+self.predict+"/model"+self.route+self.seq+self.predict+".ckpt")

    def scoreModel(self):
        print('scoreModel')
        print('Cost :',self.sesson.run(self.cost, feed_dict={self.xs:self.X_test,self.ys:self.y_test}))
        count = 0
        sum_loss = 0
        self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_test})
        self.y_test = NeuralNetModel.denormalize(self.df,self.y_test,'time_different')
        self.pred = NeuralNetModel.denormalize(self.df,self.pred,'time_different')
        for i in range(self.y_test.shape[0]):
            print('Original :',self.y_test[i],'Predicted :',self.pred[i])
            sum_loss = sum_loss + abs(self.y_test[i]-self.pred[i])
            count = i
        error_rate = sum_loss/count
        print('self.time')
        print(self.time)
        print('self.learning_rate')
        print(self.learning_rate)
        print('error_rate = ')
        print(error_rate)
        plt.plot(range(self.y_test.shape[0]),self.y_test,label="Original Data")
        plt.plot(range(self.y_test.shape[0]),self.pred,label="Predicted Data")
        plt.legend(loc='best')

    def denormalize(df,norm_data,predict):
        scl = MinMaxScaler()
        a = scl.fit_transform(df[predict].as_matrix().reshape(-1, 1))
        new = scl.inverse_transform(norm_data)
        return new

    def neural_net_model(X_data,input_dim):
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
    df = pd.read_csv('data_'+route+seq+'_'+predict+'.csv')
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

predict = '2'
learning_rate = 0.001
time = 200
#time = 10
# times = [50,100,200]
# learning_rates = [0.001,0.005,0.01]
seq = '1'
route = '11M'
NNModel = NeuralNetModel(predict, learning_rate, time, seq, route)
NNModel.train()
NNModel.scoreModel()

# t_data = [3.0888946777375796,2,21,35.08351036981856]
# result = predictResult(t_data,route,seq,predict)
# print(result)
