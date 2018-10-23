import tensorflow as tf
import numpy as np
import cv2

x = tf.placeholder(tf.float32, [1])
y = tf.placeholder(tf.float32, [1])
z = tf.placeholder(tf.float32, [8])

inputs = tf.concat([x, y, z], axis=0)
inputs = tf.stack([inputs])

inputs = tf.layers.dense(
    inputs=inputs,
    units=32,
    activation=tf.nn.relu,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

inputs = tf.layers.dense(
    inputs=inputs,
    units=64,
    activation=tf.nn.relu,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

inputs = tf.layers.dense(
    inputs=inputs,
    units=128,
    activation=tf.nn.relu,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

inputs = tf.layers.dense(
    inputs=inputs,
    units=64,
    activation=tf.nn.relu,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

inputs = tf.layers.dense(
    inputs=inputs,
    units=32,
    activation=tf.nn.sigmoid,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

pixel = tf.layers.dense(
    inputs=inputs,
    units=3,
    activation=tf.nn.relu,
    kernel_initializer=tf.variance_scaling_initializer(),
    bias_initializer=tf.zeros_initializer()
)

pixel = tf.Print(pixel, [pixel], "pixel: ")

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    image = np.array([[
        session.run(
            fetches=pixel,
            feed_dict={x: [i], y: [j], z: np.random.normal(size=[8])}
        ) for i in range(1024)
    ] for j in range(1024)])

    cv2.imshow("image", image)
    cv2.waitKey()
