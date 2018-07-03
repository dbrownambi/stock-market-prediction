import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def ann_prediction(df_train, df_test):
    # Hyper-params
    learning_rate = 0.035
    n_stocks = 5
    n_neurons_1 = 40
    n_neurons_2 = 20
    n_neurons_3 = 10
    n_neurons_4 = 5
    n_target = 1
    # Number of epochs and batch size
    epochs = 15
    batch_size = 64
    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(df_train.values)
    df_train = scaler.transform(df_train.values)
    print(df_train[0])
    df_test = scaler.transform(df_test.values)
    print(df_test[0])

    # Build input and output labels from the dataframe
    X_train = df_train[:, :5]
    y_train = df_train[:, 5]

    X_test = df_test[:, :5]
    y_test = df_test[:, 5]

    print("yoyoyo")
    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])


    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))
    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

    # Make Session
    net = tf.Session()
    # Run initializer
    net.run(tf.global_variables_initializer())

    # Setup interactive plot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_test*0.5)
    plt.show()

    for e in range(epochs):

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 5) == 0:
                # Prediction
                pred = net.run(out, feed_dict={X: X_test})
                line2.set_ydata(pred)
                # plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                plt.title('ANN vs actual prices scaled data')
                file_name = 'models/img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
                plt.savefig(file_name)
                plt.pause(0.01)
    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    y_pred = net.run(out, feed_dict={X: X_test})
    df_test[:, 5] = y_pred[0]
    y_pred = scaler.inverse_transform(df_test)
    print(len(y_pred[0]))
    print(mse_final)
    return y_pred[:, 5]









