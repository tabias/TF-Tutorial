#-*- coding: utf-8 -*-
# tensor flow
import tensorflow as tf

# training data
x_data = [1,2,3]
y_data = [1,2,3]

# w & b는 tensorflow의 변수로 처리해서 랜덤한 값을 부여함
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 그리고 y=wx+b라는 가설을 만들어냄
# 여기서 우리는 x와 y값은 w가 1이고 b가 0일때 생긴다는 것을 이미 알고 있음
hypothesis = w*x_data + b

# w & b를 찾기위해 cost를 구하는 과정임(가설값에서 y값을 제한 것에 제곱의평균)
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

# cost를 최소화하는 과정임
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
trainer = optimizer.minimize(cost)

# 모든 변수를 초기화 해주어야 오류가 나지 않음
# 다른 세션을 시작하기 전에 init을 먼저 run해주어야 함
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 2000번의 시도로 w가 1이고 b가 0이라는 사실을 컴퓨터가 알아냄
# linear regression의 실현
for step in xrange(2001):
    sess.run(trainer)
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(w), sess.run(b)