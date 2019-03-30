import tensorflow as tf
import numpy as np
import skimage
import sys


def generator(inputs, stddev):

    for _ in range(128):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.initializers.random_normal(stddev=stddev)
        )
        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True
        )
        inputs = tf.nn.sigmoid(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1,
        use_bias=False,
        kernel_initializer=tf.initializers.random_normal(stddev=stddev)
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=True
    )
    inputs = tf.nn.sigmoid(inputs)

    return inputs


def generate(image_size):

    inputs = tf.placeholder(tf.float32, [None, 3])
    outputs = generator(inputs, stddev=0.05)

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        xs = np.linspace(
            np.random.normal(0.0, 1.0),
            np.random.normal(1.0, 1.0),
            image_size
        )
        ys = np.linspace(
            np.random.normal(0.0, 1.0),
            np.random.normal(1.0, 1.0),
            image_size
        )
        z = np.random.normal(0.0, 1.0)

        image = session.run(
            fetches=outputs,
            feed_dict={inputs: [[x, y, z] for x in xs for y in ys]}
        )
        image = np.reshape(image, [image_size, image_size])
        image = image ** 2

    return image


if __name__ == "__main__":

    image = generate(int(sys.argv[1]))
    skimage.io.imsave("image.png", image)
