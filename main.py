import tensorflow as tf

from models.linear import Linear


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    print(model.variables)
