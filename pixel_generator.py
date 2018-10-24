import tensorflow as tf
import numpy as np
import cv2


def generator(inputs, stddev):

    def grow(inputs, depth, max_depth):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
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
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True,
            fused=True
        )

        inputs = tf.nn.sigmoid(inputs)

        return inputs

    # inputs = shrink(grow(inputs, 0, 3), 3, 0)

    for _ in range(128):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True,
            fused=True
        )

        inputs = tf.nn.sigmoid(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1,
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=stddev)
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=True,
        fused=True
    )

    inputs = tf.nn.sigmoid(inputs)

    return inputs


inputs = tf.placeholder(tf.float32, [None, 3])
outputs = generator(inputs, stddev=0.05)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    z = np.random.uniform(low=0.0, high=1.0)

    feed_dict = {
        inputs: [
            [x, y, z]
            for y in np.linspace(0.0, 1.0, 1024)
            for x in np.linspace(0.0, 1.0, 1024)
        ]
    }

    image = session.run(outputs, feed_dict=feed_dict)
    image = np.reshape(image, [1024, 1024, 1])
    image = image ** 1.5

    cv2.imwrite("image.png", image * 255.0)
    cv2.imshow("image", image)
    cv2.waitKey()
