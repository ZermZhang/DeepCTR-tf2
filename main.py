from functools import wraps

import tensorflow as tf

from models import linear, mlp, cnn, wide_deep, lr
from datas import load_data
from utils import config, runner, layers

# 获取模型的相关配置
CONFIG = config.Config("./conf/conf.yaml")
CONFIG_TRAIN = CONFIG.model_config

num_epoches = CONFIG_TRAIN.get('num_epoches', 3)
batch_size = CONFIG_TRAIN.get('batch_size', 100)
learning_rate = CONFIG_TRAIN.get('learning_rate', 0.01)

print("the model parameters:\n\tnum_epoches: {}\n\tbatch_size: {}\n\tlearning_rate: {}".format(num_epoches, batch_size, learning_rate))


# TODO: 调整各个模型测试用例的位置，保持main的整洁
def linear_runner():
    """
    the example runner for linear model
    """
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    # model = linear.Linear()
    model = lr.Lr()
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


def sequence_mlp_runner():
    """
    the example runner for Sequential mlp model
    """

    # the initial for model AND datas
    model = mlp.sequence_mlp()
    data_loader = load_data.MNISTLoader()

    # the initial for optimizer AND loss AND metrics name
    optimizer_func = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def loss_func(y_true, y_pred): return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred))

    model.compile(optimizer=optimizer_func, loss=loss_func, metrics=['sparse_categorical_accuracy'])

    # training
    model.fit(data_loader.train_data, data_loader.train_label, batch_size=batch_size, epochs=num_epoches)

    # evaluate
    model.evaluate(data_loader.test_data, data_loader.test_label, batch_size=batch_size)
    # loss = 4.245840030318577
    # accuracy = 0.9788

    return 0


def mlp_runner():
    """
    the example runner for mlp model
    """
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
    # loss = 3.7360587120056152
    # accuracy = 0.9728999733924866
    return 0


def mlp_runner_utils():
    """
    the example runner for mlp model use the runner utils function
    """
    model = mlp.MLP()
    data_loader = load_data.MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def loss(y_true, y_predict): return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_predict))

    # training
    model = runner.model_train(data_loader, model, loss, optimizer, batch_size=batch_size, num_epoches=num_epoches)

    # evaluate
    metrics = ['SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy']
    results = runner.model_evaluate(data_loader, model, metrics, batch_size=batch_size)

    for (name, result) in zip(metrics, results):
        print("the {} evaluate result: {}".format(name, result.result()))
    # batch 5999: loss 1.6622872352600098
    # the SparseCategoricalAccuracy evaluate result: 0.9735999703407288
    # the SparseCategoricalCrossentropy evaluate result: 0.09004498273134232
    return 0


def dnn_runner_utils():
    """
    the examples runner for DNN model use the runner utils function
    """
    model = mlp.DNN(CONFIG)
    data_loader = load_data.MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_func = layers.LossesFunc('reduce_sum_sparse_categorical_crossentropy')

    # training
    model = runner.model_train(data_loader, model, loss_func.loss, optimizer, batch_size=batch_size, num_epoches=num_epoches)

    # evaluate
    metrics = ['SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy']
    results = runner.model_evaluate(data_loader, model, metrics, batch_size=batch_size)

    for (name, result) in zip(metrics, results):
        print("the {} evaluate result: {}".format(name, result.result()))

    # batch 5999: loss 3.444500207901001
    # the SparseCategoricalAccuracy evaluate result: 0.9742000102996826
    # the SparseCategoricalCrossentropy evaluate result: 0.08227363228797913

    return 0


def cnn_runner():
    """
    the example runner for cnn model
    """
    model = cnn.CNN()
    data_loader = load_data.MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    import tensorflow_datasets as tfds

    dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
    dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
    model = tf.keras.applications.MobileNetV2(weights=None, classes=5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for e in range(num_epoches):
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                labels_pred = model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
                loss = tf.reduce_mean(loss)
                print("loss %f" % loss.numpy())
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        print(labels_pred)

    return 0


def sequence_lr_runner():
    """
    the example runner for Sequential linear model
    """
    # X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # y = tf.constant([[10.0], [20.0]])

    model = lr.Lr(CONFIG)
    data_loader = load_data.MNISTLoader()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_func = layers.LossesFunc('reduce_sum_sparse_categorical_crossentropy')

    # training
    model = runner.model_train(data_loader, model, loss_func.loss, optimizer, batch_size=batch_size, num_epoches=num_epoches)

    # evaluate
    metrics = ['SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy']
    results = runner.model_evaluate(data_loader, model, metrics, batch_size=batch_size)

    for (name, result) in zip(metrics, results):
        print("the {} evaluate result: {}".format(name, result.result()))

    # batch 49999: loss 0.5861628651618958
    # the SparseCategoricalAccuracy evaluate result: 0.9217687249183655
    # the SparseCategoricalCrossentropy evaluate result: 0.2908824384212494

    return 0


if __name__ == '__main__':
    # wide_deep.tester(CONFIG)
    sequence_lr_runner()
