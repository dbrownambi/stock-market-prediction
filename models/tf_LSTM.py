import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time


def ltsm_prediction():
    start = time.clock()
    tesla_stocks = pd.read_csv('data/DJI_5_years.csv')
    # tesla_stocks = pd.read_csv('../data/tesla_stocks.csv')

    data_to_use = tesla_stocks['Close'].values
    epochs = 200
    batch_size = 5
    learning_rate = 0.00025
    hidden_layer_size = 30
    dropout_rate = 0.98
    gradient_clip_margin = 5
    window_length = 5


    print('Total number of days in the dataset: {}'.format(len(data_to_use)))

    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(data_to_use.reshape(-1, 1))

    '''plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
    plt.title('Scaled TESLA stocks from August 2014 to August 2017')
    plt.xlabel('Days')
    plt.ylabel('Scaled value of stocks')
    plt.plot(scaled_dataset, label='Stocks data')
    plt.legend()'''
    #plt.show()


    def window_data(data, window_size):
        X = []
        y = []

        i = 0
        while (i + window_size) <= len(data) - 1:
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])

            i += 1
        assert len(X) == len(y)
        return X, y


    X, y = window_data(scaled_dataset, window_length)
    pruned_data = scaled_dataset[window_length:]
    print(len(X))
    print(len(y))

    X_train = np.array(X[:1000])
    y_train = np.array(y[:1000])

    X_test = np.array(X[1000:])
    y_test = np.array(y[1000:])


    '''X_train  = np.array(X[:500])
    y_train = np.array(y[:500])
    
    X_test = np.array(X[500:])
    y_test = np.array(y[500:])'''


    print("X_train size: {}".format(X_train.shape))
    print("y_train size: {}".format(y_train.shape))
    print("X_test size: {}".format(X_test.shape))
    print("y_test size: {}".format(y_test.shape))


    def LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout=True, dropout_rate=0.2):
        layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)

        if dropout:
            layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)

        cell = tf.contrib.rnn.MultiRNNCell([layer] * number_of_layers)

        init_state = cell.zero_state(batch_size, tf.float32)

        return cell, init_state


    def output_layer(lstm_output, in_size, out_size):
        x = lstm_output[:, -1, :]
        print(x)
        weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), name='output_layer_weights')
        bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')

        output = tf.matmul(x, weights) + bias
        return output


    def opt_loss(logits, targets, learning_rate, grad_clip_margin):
        losses = []
        for i in range(targets.get_shape()[0]):
            losses.append([(tf.pow(logits[i] - targets[i], 2))])

        loss = tf.reduce_sum(losses) / (2 * batch_size)

        # Cliping the gradient loss
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        return loss, train_optimizer


    class StockPredictionRNN(object):
        def __init__(self, learning_rate=learning_rate, batch_size=window_length, hidden_layer_size=hidden_layer_size, number_of_layers=1,
                     dropout=True, dropout_rate=dropout_rate, number_of_classes=1, gradient_clip_margin=gradient_clip_margin, window_size=window_length):
            self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1], name='input_data')
            self.targets = tf.placeholder(tf.float32, [batch_size, 1], name='targets')

            cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate)

            outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)

            self.logits = output_layer(outputs, hidden_layer_size, number_of_classes)

            self.loss, self.opt = opt_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)


    tf.reset_default_graph()
    model = StockPredictionRNN()


    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(epochs):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while (ii + batch_size) <= len(X_train):
            X_batch = X_train[ii:ii + batch_size]
            y_batch = y_train[ii:ii + batch_size]

            o, c, _ = session.run([model.logits, model.loss, model.opt],
                                  feed_dict={model.inputs: X_batch, model.targets: y_batch})

            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 10) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))


    sup =[]
    for i in range(len(traind_scores)):
        for j in range(len(traind_scores[i])):
            sup.append(traind_scores[i][j])

    tests = []
    i = 0
    while i + batch_size <= len(X_test):
        o = session.run([model.logits], feed_dict={model.inputs: X_test[i:i + batch_size]})
        i += batch_size
        tests.append(o)


    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j])


    test_results = []
    print(len(tests_new))
    tests_new_rescaled = scaler.inverse_transform(tests_new)
    for i in range(1255):
        if i >= 1000:
            test_results.append(tests_new[i-1000])
        else:
            test_results.append(None)

    '''test_results = []
    for i in range(749):
        if i >= 501:
            test_results.append(tests_new[i-501])
        else:
            test_results.append(None)'''

    end = time.clock()
    print("Time taken: ", ((end - start)), "Seconds")
    plt.figure(figsize=(16, 7))
    plt.plot(pruned_data, label='Original data')
    plt.plot(sup, label='Training data')
    plt.plot(test_results, label='Testing data')
    plt.legend()
    # plt.show()
    plt.savefig("lstm.jpg")
    session.close()
    return tests_new_rescaled
