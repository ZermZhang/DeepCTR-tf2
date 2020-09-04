import tensorflow as tf

from models import linear, mlp
from datas import load_data


def sequence_linear_runner():
    """
    the example runner for Sequential linear model
    """
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    model = linear.sequence_linear()

    model.compile(loss='mean_squared_error', optimizer='SGD')

    model.fit(X, y, epochs=100)

    print(model.variables)
    # array([[0.40425897],
    #        [1.1903621],
    #        [1.9764657]], dtype=float32)

    return 0


# TODO: try to abstract the call progress into a method
def linear_runner():
    """
    the example runner for linear model
    """
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    model = linear.Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    print(model.variables)

    # array([[0.16730985],
    #        [1.14391],
    #        [2.1205118]], dtype=float32)
    return 0


def mlp_runner():
    """
    the example runner for mlp model
    """
    # TODO: gather the config parameters into a configure file
    num_epoches = 5
    batch_size = 50
    learning_rate = 0.001

    model = mlp.MLP()
    data_loader = load_data.MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    num_batches = int(data_loader.num_train_data // batch_size * num_epoches)

    # training
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_sum(loss)
            print("batch {batch_index}: loss {loss}".format(batch_index=batch_index, loss=loss))

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    # evaluate
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)

    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)

    print("test accuracy: {}".format(sparse_categorical_accuracy.result()))
    # 0.974
    return 0


if __name__ == '__main__':
    mlp_runner()
