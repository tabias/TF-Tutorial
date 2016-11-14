#-*- coding: utf-8 -*-
# tensor flow
import tensorflow as tf

# 데이터 특징이 2개인 것(n * 1 행렬로 표현)
x = [[1, 1, 1, 1, 1],
     [1, 0, 3, 0, 5],
     [0, 2, 0, 4, 0]]

y = [1, 2, 3, 4, 5]


# 플레이스홀더 처리
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# 변수 랜덤값 설정(1*n 행렬로 설정, b를 없애기 위해 [1, 1, 1, 1, 1]추가함)
w = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))


# 가설 공식 적용(특징이 여러개이기 때문에 w * X의 행렬의 곱셈을 해줘야함)
hypothesis = tf.matmul(w, X)


# 비용 계산, gradientdescent algorithm
cost = tf.reduce_mean(tf.square(hypothesis-y))
LR = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(LR)
train = optimizer.minimize(cost)


session = tf.Session()
init = tf.initialize_all_variables()
session.run(init)


# session print & get w, b value
for step in xrange(2001):
    session.run(train, feed_dict={X: x, Y: y})
    if(step % 20 ==0):
        print session.run(w)


# example data
x2 = [[1, 1, 1, 1, 1],
      [5, 4, 3, 2, 1],
      [5, 4, 3, 2, 1]]


# 결과예측(5개의 트레이닝 셋을 연습시켜서 w,b 값을 얻어냈고, 이제 이것을 바탕으로 예측하기 위해 가설값으로 예측해본다.(1, 특징1, 특징2)로
#print session.run(hypothesis, feed_dict={X: x2})
print session.run(hypothesis, feed_dict={X: [[1], [2], [2]]})