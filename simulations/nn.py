import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import sys

HERE_PATH = os.path.dirname(__file__)
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

class NeuralRegressor():
    def __init__(self):
        self.build_model()
        self.checkpoint_path = os.path.join(root_path, 'checkpoints', 'network')
        NUM_THREADS = 4
        self.config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
        self.session = tf.Session(config=self.config)

    def build_model(self):
        tf.reset_default_graph()
        
        # build computation graph
        #  data placeholders
        self.X= tf.placeholder(tf.float32, shape=[None, 37], name='X')
        self.Y= tf.placeholder(tf.float32, shape=[None, 1], name='Y')
        self.keep_prob = tf.placeholder(tf.float32)
        
        # layer 1
        W_init = tf.truncated_normal(shape=[37, 50], stddev=0.01,mean=0.0, name='W_init')
        W = tf.Variable(W_init, dtype=tf.float32, name='W')
        bias_init = tf.zeros([50], dtype=tf.float32)
        bias = tf.Variable(bias_init, dtype=tf.float32)
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(self.X, W) + bias),
                                keep_prob=self.keep_prob)
        
        # layer 2
        W2_init = tf.random_normal(shape=[50, 7], stddev=0.01, name='W2_init')
        W2 = tf.Variable(W2_init, dtype=tf.float32, name='W2')
        bias2_init = tf.zeros([7], dtype=tf.float32)
        bias2 = tf.Variable(bias2_init, dtype=tf.float32)
        layer2 = tf.nn.sigmoid(tf.matmul(layer_1, W2) + bias2)
        
        #layer 3 and linear regression
        W3_init = tf.random_normal(shape=[7, 1], stddev=0.01, name='W3_init')
        W3 = tf.Variable(W3_init, dtype=tf.float32, name='W3')
        bias3_init = tf.zeros([1], dtype=tf.float32)
        bias3 = tf.Variable(bias3_init, dtype=tf.float32)
        self.y_pred = (tf.matmul(layer2,W3) + bias3)
        # loss function
        self.loss = tf.reduce_mean(tf.square(self.Y- self.y_pred))
        # optimizer
        self.step = tf.train.AdamOptimizer(learning_rate=.005).minimize(self.loss)

    def reset(self):
        init = tf.global_variables_initializer()
        self.session.run(init)
        

    def train_model(self, train, validation):
        
        batch_size = 100
        epochs = 300
        
        saver = tf.train.Saver()
        iterations_per_epoch = len(train.X) // batch_size
        best_loss = np.infty
        self.epoch = []
        self.training_loss = []
        self.validation_loss = []
        
       
        self.session.run(tf.global_variables_initializer())
        for epcoch in tqdm(range(epochs)):
            self.epoch.append(epcoch)
            for i in range(iterations_per_epoch):
                train_losses =[]
                batchx, batchy = train.next_batch(batch_size)
                batchy = batchy.reshape(-1, 1)
                
                _, batch_loss = self.session.run([self.step, self.loss], 
                         feed_dict = {self.X:batchx, self.Y:batchy, self.keep_prob:0.8})
                train_losses.append(batch_loss)
                
            self.training_loss.append(np.mean(train_losses))
            
            yp, val_loss = self.session.run([self.y_pred, self.loss],
                                      feed_dict={self.X:validation.X,
                                                 self.Y:validation.Y.reshape(-1, 1),
                                                 self.keep_prob:1.0})
            self.validation_loss.append(val_loss)
            
        
            if val_loss< best_loss and epcoch%20==0:
                print('New best validation loss {}'.format(val_loss))
                best_loss = val_loss
                saver.save(self.session, self.checkpoint_path)
                time.sleep(0.1)
                
        saver.restore(self.session, self.checkpoint_path)
    
        print ('Finised with validation loss {}'.format(val_loss))
        print ('Best validation loss {}'.format(best_loss))
    
    def test_model(self, test):
        
        yp, test_loss = self.session.run([self.y_pred, self.loss],
                                 feed_dict={self.X:test.X,
                                    self.Y:test.Y.reshape(-1, 1),
                                    self.keep_prob:1.0})
        
        print ('Test Loss (MSE) {}'.format(test_loss))
        print ('Test RMSE {}'.format(np.sqrt(test_loss)))
        
        plt.scatter(test.Y, yp, alpha=0.3, c='g')
        plt.title('Prediction of yield from test set')
        plt.xlabel('True Yield')
        plt.ylabel('Predicted Yield')
        plt.show()
        path = os.path.join(root_path,'figures', 'Testset.pdf')
        plt.savefig(path)
        plt.close()
        
        return yp, test_loss
            
    
    def predict(self, data):


            
        yp, loss = self.session.run([self.y_pred, self.loss],
                                     feed_dict={self.X:data.X,
                                        self.Y:data.Y.reshape(-1, 1),
                                        self.keep_prob:1.0})
    
        print ('Prediction Mean Squared Error Loss = {}'.format(loss))
        print ('Prediction Standard Deviation = {}'.format(np.std(yp)))
        
        return yp, loss
    
    def training_stats(self):
        plt.plot(self.epoch, self.training_loss, c='b', label='Training loss')
        plt.plot(self.epoch, self.validation_loss, c='g', label='Talidations loss')
        plt.title('Training of NN')
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig('Training.pdf')
        
