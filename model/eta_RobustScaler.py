import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from sklearn.externals import joblib


class NeuralNetModel():

    features = []

    drop_features = []

    predict = ''

    def __init__(self,drop_features,predict):
        print('start __init__')
        self.drop_features = drop_features
        self.predict = predict
        self.sesson = tf.Session()
        self.graph = tf.Graph()
        df = pd.read_csv('data_v3.csv')
        self.df = df.drop(['WeekDay','s0','s1','s2','s4','time_different'],axis=1)
        self.df_train = self.df[:1200]
        self.df_test = self.df[1200:]
        self.scaler = RobustScaler()
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
        self.train = tf.train.AdamOptimizer(0.0005).minimize(self.cost)
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
        for i in range(50):
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
            print('Epoch :',i,'Cost :',c_t[i])


    def scoreModel(self):
        print('scoreModel')
        print('Cost :',self.sesson.run(self.cost, feed_dict={self.xs:self.X_test,self.ys:self.y_test}))
        count = 0
        sum_loss = 0
        self.pred = self.sesson.run(self.output, feed_dict={self.xs:self.X_test})
        self.y_test = NeuralNetModel.denormalize(self.df,self.y_test)
        self.pred = NeuralNetModel.denormalize(self.df,self.pred)
        for i in range(self.y_test.shape[0]):
            print('Original :',self.y_test[i],'Predicted :',self.pred[i])
            sum_loss = sum_loss + abs(self.y_test[i]-self.pred[i])
            count = i
        error_rate = sum_loss/count
        plt.plot(range(self.y_test.shape[0]),self.y_test,label="Original Data")
        plt.plot(range(self.y_test.shape[0]),self.pred,label="Predicted Data")
        plt.legend(loc='best')
        print('error_rate = ')
        print(error_rate)


    def predictResult(self,input):
        scaler = RobustScaler()
        b = scaler.fit(self.df.drop(['s3'],axis=1).as_matrix())
        df = pd.DataFrame([input],columns = ["distance", "Hour", "v"])
        df_input = scaler.transform(df.as_matrix())
        pred = self.sesson.run(self.output, feed_dict={self.xs:df_input})
        pred = NeuralNetModel.denormalize(self.df,pred)
        return pred

    def denormalize(df,norm_data):
        # df = df['s3'].values.reshape(-1,1)
        # norm_data = norm_data.reshape(-1,1)
        scl = RobustScaler()
        a = scl.fit(df['s3'].as_matrix().reshape(-1, 1))
        new = scl.inverse_transform(norm_data)
        return new

    def neural_net_model(X_data,input_dim):
        W_1 = tf.Variable(tf.random_uniform([input_dim,100]))
        b_1 = tf.Variable(tf.zeros([100]))
        layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
        #layer_1 = tf.nn.tanh(layer_1)
        layer_1 = tf.nn.relu(layer_1)

        W_2 = tf.Variable(tf.random_uniform([100,20]))
        b_2 = tf.Variable(tf.zeros([20]))
        layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
        #layer_2 = tf.nn.tanh(layer_2)
        layer_2 = tf.nn.relu(layer_2)

        W_O = tf.Variable(tf.random_uniform([20,1]))
        b_O = tf.Variable(tf.zeros([1]))
        output = tf.add(tf.matmul(layer_2,W_O), b_O)

        return output,W_O


drop_features = ['WeekDay','s0','s1','s2','s3','time_different']

predict = 's4'

NNModel = NeuralNetModel(drop_features, predict)

NNModel.train()

NNModel.scoreModel()

input = [3.0888946777375796,21,35.08351036981856]

result = NNModel.predictResult(input)

print('eta = ')
print(result)

print('finish the model')
