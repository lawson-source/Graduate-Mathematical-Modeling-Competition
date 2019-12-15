import tensorflow as tf
import data_preprocess
import numpy as np

xdata,ydata=data_preprocess.main()

from sklearn.model_selection import train_test_split
partial_train_data, val_data,partial_train_targets, val_targets = train_test_split(xdata, ydata, random_state=0,test_size=0.2)
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "./model5")
    graph = tf.get_default_graph()

    x0 = sess.graph.get_tensor_by_name('myInput:0')
    y = sess.graph.get_tensor_by_name('myOutput:0')

    scores = sess.run(y, feed_dict={x0: val_data}, )
    y=np.array(scores*10-103)
    ypre=val_targets*10-103
print((((y.ravel()-ypre)**2).mean())**0.5)
