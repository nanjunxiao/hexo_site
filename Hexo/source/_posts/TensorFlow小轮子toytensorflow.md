title: TensorFlow小轮子toytensorflow
date: 2018-06-23 18:32:42
categories: 机器学习
tags: [机器学习,TensorFlow]
---

## 背景与目的
本着知其然知其所以然的原则，对TensorFlow的核心特性进行探索，希望通过代码落地方式加强认知与备忘。toytensorflow是对TensorFlow python API模拟的玩具小轮子，包括DAG、惰性求值、链式法则、自动求偏导、前向/后项算法等特性。
github地址：https://github.com/nanjunxiao/toytensorflow
## 抽象四元素
1.operation
操作符、Variable、Constant、Placeholder统一抽象为operation。
2.graph
有向无环图DAG
3.session
会话，sess.run(op)才真正计算，惰性求值
4.optimizer
优化算子，比如梯度下降
## 实现notes
以linear regression为例，loss = reduce_mean(square(matmul(X,W)+b - Y) )，构建的DAG如下图所示，实线表示前向计算，虚线表示BP反向传播
![](../../../../img/toytensorflow/DAG.jpeg)
1.operation：为了支持向量化表达，操作符包括matmul等矩阵操作及求导
2.graph：通过邻接链表构建DAG，singleton实现default_graph
3.session：sess.run(op)才真正计算，DFS递归求值
4.optimizer：目前只实现了GradientDescentOptimizer
5.sess.run(train_op)时启动BP反向传播，为避免重复计算，采用BFS实现链式求导
## 例子及效果
以linear regression为例，左图是TensorFlow结果，右图是toytensorflow结果，结果是一致的。
```python
import toytensorflow as tf
import numpy as np

#real data
np.random.seed(1) #for the same data
x_data = np.float32(np.random.rand(2,100) )
y_data = np.dot([0.1,0.2], x_data) + 0.3
#ops & DAG
W = tf.Variable([[0.0,0.0] ], name='weight')
b = tf.Variable(0.0, name='bias')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
predict = tf.matmul(W, X) + b

loss = tf.reduce_mean(tf.square(predict - Y) )
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

init = tf.initialize_all_variables()
#run
feed_dict = {X:x_data, Y:y_data}
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):
        mse = sess.run(loss, feed_dict=feed_dict)
        print 'epoch: {}, mse: {}, w: {}, b: {}'.format(epoch, mse, sess.run(W), sess.run(b))
        sess.run(train_op, feed_dict=feed_dict)
    w_value = sess.run(W, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('W: {}, b: {}'.format(w_value, b_value) )
```
![](../../../../img/toytensorflow/linearregression_result.png)
## TODO
1.添加更多的操作符
2.添加更多的优化算法，比如momentum/adam/adagrad
3.丰富更多例子，比如logisticregression/mlp
