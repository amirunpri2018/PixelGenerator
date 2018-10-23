import tensorflow as tf
import numpy as np
import cv2

inputs = tf.placeholder(tf.float32, [None, 10])


def grow(inputs, depth, max_depth):

    inputs = tf.layers.dense(
        inputs=inputs,
        units=128,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(distribution="uniform")
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=True,
        fused=True
    )

    inputs = tf.nn.sigmoid(inputs)

    return inputs if depth == max_depth else [grow(inputs, depth + 1, max_depth) for _ in range(2)]


def shrink(inputs_seq, depth, min_depth):

    inputs = tf.concat(inputs_seq, axis=1) if depth == min_depth else tf.concat(
        [shrink(inputs, depth - 1, min_depth) for inputs in inputs_seq], axis=1)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=128,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(distribution="uniform")
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=True,
        fused=True
    )

    inputs = tf.nn.sigmoid(inputs)

    return inputs


outputs = shrink(grow(inputs, 0, 3), 3, 0)

outputs = tf.layers.dense(
    inputs=outputs,
    units=1,
    use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(distribution="uniform")
)

outputs = tf.layers.batch_normalization(
    inputs=outputs,
    axis=-1,
    training=True,
    fused=True
)

outputs = tf.nn.sigmoid(outputs)

pixel = tf.Print(outputs, [outputs], "pixel: ")


def scale(input, input_min, input_max, output_min, output_max):

    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)


with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    z = np.random.uniform(low=0.0, high=1.0, size=8)

    feed_dict = {inputs: [
        np.concatenate([[x, y], z])
        for y in np.linspace(0.0, 1.0, 1024)
        for x in np.linspace(0.0, 1.0, 1024)
    ]}

    image = session.run(pixel, feed_dict=feed_dict).reshape([1024, 1024, 1])

    cv2.imshow("image", image)

    cv2.waitKey()
