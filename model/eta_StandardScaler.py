import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.externals import joblib


class NeuralNetModel():

    features = []

    drop_features = []

    predict = ''

    learning_rate = 0

    time = 0

    def __init__(self,drop_features,predict, learning_rate, time):
        print('start __init__')
        self.learning_rate = learning_rate
        self.time = time
        self.drop_features = drop_features
        self.predict = predict
        self.sesson = tf.Session()
        self.graph = tf.Graph()
        df = pd.read_csv('data_v3.csv')
        self.df = df.drop(['WeekDay','s0','s1','s2','s4','time_different'],axis=1)
        self.df_train = self.df[:1200]
        self.df_test = self.df[1200:]
        self.scaler = StandardScaler()
        b = self.scaler.fit(self.df.drop(['s3'],axis=1).as_matrix())
        self.X_train = self.scaler.transform(self.df_train.drop(['s3'],axis=1).as_matrix())
        self.X_test = self.scaler.transform(self.df_test.drop(['s3'],axis=1).as_matrix())


        c = self.scaler.fit(self.df['s3'].as_matrix().reshape(-1, 1))
        self.y_train = self.scaler.transform(self.df_train['s3'].as_matrix().reshape(-1, 1))
        # self.X_test = self.scaler.transform(self.df_test.drop(['s3'],axis=1).as_matrix())
        self.y_test = self.scaler.transform(self.df_test['s3'].as_matrix().reshape(-1, 1))

        self.xs = tf.placeholder("float")
        self.ys = tf.placeholder("float")

        print('end __init__')

    def train(self):
        self.output,self.W_O = NeuralNetModel.neural_net_model(self.xs,3)
        self.cost = tf.reduce_mean(tf.square(self.output-self.ys))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.correct_pred = tf.argmax(self.output, 1)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))
        c_t = []
        c_test = []
        self.sesson.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        self.y_t = NeuralNetModel.denormalize(self.df,self.y_train)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        plt.xlabel('Records')
        plt.ylabel('Time')
        plt.title('Minibus Eta prediction')
        self.ax.plot(range(len(self.y_train)), self.y_t,label='Original')
        plt.ion()
        for i in range(self.time):
            for j in range(self.X_train.shape[0]):
                self.sesson.run([self.cost,self.train],feed_dict={self.xs:self.X_train[j,:].reshape(1,3), self.ys:self.y_train[j]})

            try:
                self.ax.lines.remove(lines[0])
            except Exception:
                pass
            self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_train})
            self.pred = NeuralNetModel.denormalize(self.df,self.pred)
            lines = self.ax.plot(range(len(self.y_train)), self.pred,'r-',label='Prediction')
            plt.legend(loc='best')
            plt.pause(0.1)

            c_t.append(self.sesson.run(self.cost, feed_dict={self.xs:self.X_train,self.ys:self.y_train}))
            c_test.append(self.sesson.run(self.cost, feed_dict={self.xs:self.X_test,self.ys:self.y_test}))


    def scoreModel(self):
        print('scoreModel')
        print('Cost :',self.sesson.run(self.cost, feed_dict={self.xs:self.X_test,self.ys:self.y_test}))
        count = 0
        sum_loss = 0
        self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_test})
        self.y_test = NeuralNetModel.denormalize(self.df,self.y_test)
        self.pred = NeuralNetModel.denormalize(self.df,self.pred)
        for i in range(self.y_test.shape[0]):
            sum_loss = sum_loss + abs(self.y_test[i]-self.pred[i])
            count = i
        error_rate = sum_loss/count
        plt.plot(range(self.y_test.shape[0]),self.y_test,label="Original Data")
        plt.plot(range(self.y_test.shape[0]),self.pred,label="Predicted Data")
        plt.legend(loc='best')
        print('time = ')
        print(self.time)
        print('learning_rate = ')
        print(self.learning_rate)
        print('error_rate = ')
        print(error_rate)


    def predictResult(self,input):
        df = pd.DataFrame([input],columns = ["distance", "Hour", "v"])
        self.scaler = StandardScaler()
        b = self.scaler.fit(self.df.drop(['s3'],axis=1).as_matrix())
        df_input = self.scaler.transform(df.as_matrix())

        # print("df_input")
        # print(df_input)
        # # input_pre = self.sesson.run(self.output, feed_dict={self.xs:df_input})
        # NeuralNetModel.neural_net_model(self.xs,3)
        # # self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_test})
        # # self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_train})
        # denormalize_pred = NeuralNetModel.denormalize(self.df_train,input_pre)
        # print('denormalize_pred')
        # print(denormalize_pred)
        # return denormalize_pre
        return 1

    def denormalize(df,norm_data):
        # df = df['s3'].values.reshape(-1,1)
        # norm_data = norm_data.reshape(-1,1)
        scl = StandardScaler()
        a = scl.fit(df['s3'].as_matrix().reshape(-1, 1))
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


drop_features = ['WeekDay','s0','s1','s2','s4','time_different']

predict = 's3'

learning_rates = [0.001,0.005,0.01]

counts = [50, 100, 200, 400]

NNModel = NeuralNetModel(drop_features, predict, 0.005, 200)
NNModel.train()
NNModel.scoreModel()

# input = [3.0888946777375796,21,35.08351036981856]
#
# result = NNModel.predictResult(input)

print(result)

print('finish the model')
