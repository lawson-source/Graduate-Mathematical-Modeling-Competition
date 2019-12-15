
import tensorflow as tf
import pandas as pd


batch_size = 40


import data_preprocess

xdata,ydata=data_preprocess.main()

from sklearn.model_selection import train_test_split
partial_train_data, val_data,partial_train_targets, val_targets = train_test_split(xdata, ydata, random_state=0,test_size=0.2)

def train(bias):
    # 初始化结构
    x = tf.placeholder("float", [None, 8], name='myInput')
    y_ = tf.placeholder("float", [None, 1], name='y-input')

    W1 = tf.Variable(tf.random_uniform([8, 4], -0.5 + bias, 0.5 + bias))
    b1 = tf.Variable(tf.random_uniform([4], -0.5 + bias, 0.5 + bias))
    u1 = tf.matmul(x, W1) + b1
    y1 = tf.nn.sigmoid(u1)
    #    y1=u1
    W2 = tf.Variable(tf.random_uniform([4, 1], -0.5 + bias, 0.5 + bias))
    b2 = tf.Variable(tf.random_uniform([1], -0.5 + bias, 0.5 + bias))
    y = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)
    # 设置训练方式
    mse = tf.reduce_mean((tf.square(y - y_)))  # mse
    # train_step = tf.train.GradientDescentOptimizer(0.02).minimize(mse)
    train_step = tf.train.AdamOptimizer(0.001).minimize(mse)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.identity(y, name="myOutput")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 开始训练
        STEPS = 200000000
        note=1
        for i in range(0,STEPS,10):

            start = (i)
            end = start + 10
            sess.run(train_step, feed_dict={x: partial_train_data[start:end],
                                            y_: pd.DataFrame({'y': partial_train_targets[start:end]})})
            if i % 10000 == 0:
                total_loss = sess.run(mse, feed_dict={x: val_data, y_: pd.DataFrame({'y': val_targets})})
                print("After %d training step(s), loss on all data is %g" % (i, total_loss))
                if abs(note-total_loss)<=0.000000001:
                    break
                note=total_loss



        print('bias range from [%.1f,%.1f],test accuracy'
              % (bias - 0.5, bias + 0.5),
              sess.run(accuracy, feed_dict={x: val_data, y_: pd.DataFrame({'y': val_targets})}))
        tf.saved_model.simple_save(sess, "./model5", inputs={"myInput": x}, outputs={"myOutput": y})



if __name__ == '__main__':
    init_bias = -0.1
    train(init_bias)


