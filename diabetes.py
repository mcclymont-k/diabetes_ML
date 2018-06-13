import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.nan)

diabetes_data = pd.read_csv('pima-indians-diabetes.csv')


# Seperate into features and labels + remove unneccesary columns
features_data = diabetes_data.drop('Class', axis=1)
features_data = features_data.drop('Group', axis=1)

label = []
label_count = 0
diab_class = diabetes_data['Class']
for i in diab_class:
    if i == 1:
        label.append([0, 1])
        label_count+=1
    else:
        label.append([1, 0])

# Normalize the data - labels is already ready to go.
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree', 'Age']

features_data[cols_to_norm] = features_data[cols_to_norm].apply(
    lambda x: (x-x.min())/(x.max() - x.min()))

# Create a train test split

features_data_array = np.array(features_data).astype('float32')
label_array = np.array(label)


x_train, x_test, y_train, y_test = train_test_split(features_data_array,
                                                    label_array, test_size=0.3)

print(x_train)
print(y_train)

x = tf.placeholder('float', [None, 8])
y = tf.placeholder('float')

# Create the neural network architechture

n_nodes_hidden_layer_1 = 10
n_nodes_hidden_layer_2 = 10

n_classes = 2
batch_size = 20

def neural_network_model(data):
    hidden_layer_1 = {
        'weights': tf.Variable(tf.truncated_normal([8, n_nodes_hidden_layer_1], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1,shape=[n_nodes_hidden_layer_1]))
    }

    hidden_layer_2 = {
        'weights': tf.Variable(tf.truncated_normal([n_nodes_hidden_layer_1, n_nodes_hidden_layer_2], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1,shape=[n_nodes_hidden_layer_2]))
    }

    output_layer = {
        'weights': tf.Variable(tf.truncated_normal([n_nodes_hidden_layer_2, n_classes], stddev=0.1)),
        'biases': tf.Variable(tf.constant(0.1,shape=[n_classes]))
    }

    layer1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer2 = tf.nn.sigmoid(layer2)

    output = tf.add(tf.matmul(layer2, output_layer['weights']), output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # the problem with the second cost variable is that it only works for models with
    # multiple outputs
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 1000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
    # Determines how many data points in each batch
            for c in range(int(len(x_train)/batch_size)):
                x_batch = []
                y_batch = []
                for i in range(10):
                    epoch_x = x_train[(i + (c*10))]
                    epoch_y = y_train[(i + (c*10))]
                    x_batch.append(epoch_x)
                    y_batch.append([epoch_y])
                a, b = sess.run([optimizer, cost], feed_dict={x:x_batch, y:y_batch})
                epoch_loss += b
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print ('Accuracy: ', accuracy.eval({x:x_test, y: y_test}))

train_neural_network(x)
