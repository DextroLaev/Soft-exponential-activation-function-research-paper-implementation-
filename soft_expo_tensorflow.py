import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

class Dense_layer(tf.keras.layers.Layer):
    def __init__(self, output_neurons):
        super(Dense_layer, self).__init__()
        self.units = output_neurons

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='ones',
                                 trainable=True)
        
    def call(self, inputs):
        inputs = tf.cast(inputs,dtype=tf.float32)
        dot_product = tf.matmul(inputs, self.w)+self.b
        return dot_product

class soft_exponential(tf.keras.layers.Layer):
    def __init__(self,alpha=1):
        super(soft_exponential,self).__init__()
        self.alpha_val = tf.keras.backend.cast_to_floatx(alpha)

    def build(self,input_shape):
        input_shape = input_shape[1:]
        self.alpha = self.add_weight(shape=(input_shape[-1],),
            initializer='ones',trainable=True)
        self.trainable_weight = [self.alpha]

    def get_alpha_gt0(self,inputs,alpha):
        return alpha+(tf.math.exp(alpha*inputs)-1.)/alpha

    def get_alpha_lt0(self,inputs,alpha):
        return -(1/alpha) * (tf.math.asinh(1-alpha*(inputs+alpha))) 

    def call(self,x):
        return tf.keras.backend.switch(self.alpha > 0, self.get_alpha_gt0(x, self.alpha), tf.keras.backend.switch(self.alpha < 0, self.get_alpha_lt0(x, self.alpha), x))

class Neural_nets:

    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = tf.cast(train_data/255, dtype=tf.float32)
        self.train_label = tf.cast(train_label, dtype=tf.float32)
        self.test_data = tf.cast(test_data/255, dtype=tf.float32)
        self.test_label = tf.cast(test_label, dtype=tf.float32)
        self.epochs = 20
        self.lr = 0.0005
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 150
        self.batch_no = 0
        self.init_model_structure()

    def init_model_structure(self):    
        self.flatten_data = tf.keras.layers.Flatten()
        self.hd1 = Dense_layer(128)
        self.soft1 = soft_exponential()
        self.hd2 = Dense_layer(84)
        self.soft2 = soft_exponential()
        self.hd3 = tf.keras.layers.Dense(10, activation='softmax')

    @tf.function    
    def model(self,data):
        model_input = tf.reshape(data, shape=[-1, 28, 28])
        flatten_data = self.flatten_data(model_input)

        hd1 = self.hd1(flatten_data)
        soft1 = self.soft1(hd1)

        hd2 = self.hd2(soft1)
        soft2 = self.soft2(hd2)

        hd3 = self.hd3(soft2)
        return hd3

    def get_weights(self):
        weights1 = self.hd1.weights
        alpha1 = self.soft1.weights
        weights2 = self.hd2.weights
        alpha2 = self.soft2.weights
        weights3 = self.hd3.weights
        self.weights = [weights3,alpha2, weights2,alpha1, weights1]
        return self.weights

    @tf.function    
    def accuracy(self, pred, label):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, 1), tf.argmax(pred, 1)), dtype=tf.float32))

    def train(self):
        for i in range(self.epochs):
            print('Epochs = {}/{}'.format(i+1, self.epochs))
            for j in range(60000//self.batch_size):

                batch_data, batch_label = self.get_next_batch()
                up_weights = self.get_weights()
                def loss(): return self.loss(batch_label,self.model(batch_data))
                self.optimizer.minimize(loss,var_list=up_weights)

                tr_loss = self.loss(self.train_label,self.model(self.train_data))
                tt_loss = self.loss(self.test_label,self.model(self.test_data))
                tt_acc = self.accuracy(self.model(self.test_data),self.test_label)
                tr_acc = self.accuracy(self.model(self.train_data),self.train_label)
                print('\r batch = {}/{} ,train_loss = {}, test_loss = {}, train_acc = {}, test_acc = {}'.format((j+1)*self.batch_size,60000,tr_loss,tt_loss,tr_acc,tt_acc),end='')
                sys.stdout.flush()
            print()
        print()

    def get_next_batch(self):
        batch_data = self.train_data[self.batch_no:self.batch_no+self.batch_size]
        batch_label = self.train_label[self.batch_no:self.batch_no+self.batch_size]
        self.batch_no = (
            self.batch_no+self.batch_size) % (60000-self.batch_size)
        return batch_data, batch_label        

if __name__ == '__main__':
    (train_data, train_label), (test_data,
                                test_label) = tf.keras.datasets.mnist.load_data()
    train_label = tf.squeeze(tf.one_hot(train_label, depth=10))
    test_label = tf.squeeze(tf.one_hot(test_label, depth=10))
    nn = Neural_nets(train_data, train_label, test_data, test_label)
    nn.train()
