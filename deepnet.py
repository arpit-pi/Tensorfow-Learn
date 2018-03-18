import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784]) #height , width*length
y = tf.placeholder('float')

def neural_net(data):
    hidden_layer_1 = { 'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer =   { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']) , hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']) , hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']) , hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output =  tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    return output

def train_model(x):
    prediction = neural_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epochs in range(0,hm_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                ep_x , ep_y = mnist.train.next_batch(batch_size)
                i, c = sess.run([optimizer,cost],feed_dict = {x:ep_x,y:ep_y})
                epoch_loss += c
            print('epoch_completed ,loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy=',accuracy)

train_model(x)
