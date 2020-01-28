title: Actor-Critic推导及示例
date: 2018-07-01 14:34:48
categories: 机器学习
tags: [强化学习,Actor-Critic]
---
Actor-Critic算法是一种结合**策略梯度policy gradient**和**时序差分学习TD learning**的强化学习方法。其中actor（演员）是指策略函数π(s, a)，即学习一个策略来得到尽量高的回报；critic（评论员）是指值函数Vϕ(s)，对当前策略的值函数进行估计，即评估actor的好坏。然后交替学习至收敛。借助于值函数，actor-critic 算法可以进行单步更新参数，不需要等到回合结合才进行更新。
## 数学推导
![](../../../../img/Actor-Critic.jpeg)
## 伪代码示例
```python
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
    	"""
    	Critic网络结构，输出V(s),略
    	"""
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_next - self.v
            self.loss = tf.square(self.td_error) # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
```
```python
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
    	"""
    	Actor网络结构，输出p(a|s),略
    	"""
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)#minimize(-exp_v) = maximize(exp_v)
     def choose_action(self, s):
         pass
```
## 应用场景
离线没有足够label数据时，无法进行监督学习，可考虑通过线上try-and-error，进行Actor-Critic强化学习，Reward恒正也ok。
