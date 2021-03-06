import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

df = pd.read_csv('data_v3.csv')

# Line 9 to 17 is for preprocessing and saving the dataset downloaded from yahoo website
# It is not necessary to run these codes if you are using provided dataset in repo.
# If you want to use your own dataset downloaded from yahoo then first run these with commented rest.
# After that you can run as usual provided.

# df = df.drop(['Date'],axis=1)
# for col in df:
#     for i,item in enumerate(df[col]):
#         if item=='null':
#             df[col][i] = np.nan
# df = df.dropna(inplace=False)
# for col in df:
#     print(df[col].isnull().sum())
# df.to_csv('yah.csv',index=False)

df = df.drop(['WeekDay','s0','s3','s2','s4','time_different'],axis=1)




df_train = df[:1200]
df_test = df[1200:]
scaler = MinMaxScaler()

X_train = scaler.fit_transform(df_train.drop(['s1'],axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['s1'].as_matrix().reshape(-1, 1))
#y_train = df_train['Close'].as_matrix()

X_test = scaler.fit_transform(df_test.drop(['s1'],axis=1).as_matrix())
y_test = scaler.fit_transform(df_test['s1'].as_matrix().reshape(-1, 1))
#y_test = df_test['Close'].as_matrix()
print(X_train.shape)
print(np.max(y_test),np.max(y_train),np.min(y_test),np.min(y_train))

def denormalize(df,norm_data):
    df = df['s1'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
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

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output,W_O = neural_net_model(xs,3)

cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.AdamOptimizer(0.0005).minimize(cost)
#train = tf.train.GradientDescentOptimizer()

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

c_t = []
c_test = []


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    y_t = denormalize(df_train,y_train)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('Records')
    plt.ylabel('Time')
    plt.title('Minibus Eta prediction')
    ax.plot(range(len(y_train)), y_t,label='Original')
    plt.ion()

    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        #sess.run([cost,train],feed_dict={xs:X_train, ys:y_train})
        for j in range(X_train.shape[0]):
            sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape(1,3), ys:y_train[j]})

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        pred = sess.run(output, feed_dict={xs:X_train})
        pred = denormalize(df_train,pred)
        lines = ax.plot(range(len(y_train)), pred,'r-',label='Prediction')
        plt.legend(loc='best')
        plt.pause(0.1)

        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])

    pred = sess.run(output, feed_dict={xs:X_test})
    # for i in range(y_test.shape[0]):
    #     print('Original :',y_test[i],'Predicted :',pred[i])

    #plt.plot(range(50),c_t)
    #plt.plot(range(50),c_test)
    #plt.show()

    print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    y_test = denormalize(df_test,y_test)
    pred = denormalize(df_test,pred)
    sum_loss = 0
    count = 0
    for i in range(y_test.shape[0]):
        print('Original :',y_test[i],'Predicted :',pred[i])
        sum_loss = sum_loss + abs(y_test[i]-pred[i])
        count = i
    error_rate = sum_loss/count
    print('Sum Loss = ',sum_loss)
    print('Error rate = ',error_rate)

    plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    plt.legend(loc='best')
    """plt.ylabel('Stock Value')
    plt.xlabel('Days')
    plt.title('Stock Market Nifty')"""
    #plt.show()
    if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() + '/yahoo_dataset.ckpt')
        print('Model Saved')
