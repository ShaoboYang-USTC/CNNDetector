import tensorflow as tf

from config.config import Config
from reader.reader import Reader
from tflib import layers
from tflib.models import Model


class CNN(object):
    def __init__(self):
        self.config = Config()
        self.reader = Reader()
        self.layer = self.setup_layer()
        self.loss = self.setup_loss()
        self.metrics = self.setup_metrics()
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.config.root + '/event_detect/summary/cnn')

    def setup_layer(self):
        layer = dict()
        layer['target'] = tf.placeholder(tf.int32, shape=[None], name='target')
        layer['input'] = tf.placeholder(tf.float32,
                                        shape=[None, None, self.config.winsize, 3],
                                        name='input')
        layer['conv1'] = layers.conv(layer['input'],
                                     filter=[3, 3, 3, 8],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv1')
        layer['pooling1'] = layers.pool(layer['conv1'],
                                        ksize=[1, 1, 3, 1],
                                        strides=[1, 1, 3, 1],
                                        padding='SAME',
                                        pool_func=tf.nn.max_pool,
                                        name='pooling1')
        layer['conv2'] = layers.conv(layer['pooling1'],
                                     filter=[3, 3, 8, 16],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv2')
        layer['pooling2'] = layers.pool(layer['conv2'],
                                        ksize=[1, 1, 3, 1],
                                        strides=[1, 1, 3, 1],
                                        padding='SAME',
                                        pool_func=tf.nn.max_pool,
                                        name='pooling2')
        layer['conv3'] = layers.conv(layer['pooling2'],
                                     filter=[3, 3, 16, 32],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv3')
        layer['pooling3'] = layers.pool(layer['conv3'],
                                        ksize=[1, 1, 3, 1],
                                        strides=[1, 1, 3, 1],
                                        padding='SAME',
                                        pool_func=tf.nn.max_pool,
                                        name='pooling3')
        layer['conv4'] = layers.conv(layer['pooling3'],
                                     filter=[1, 3, 32, 32],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv4')
        layer['pooling4'] = layers.pool(layer['conv4'],
                                        ksize=[1, 1, 3, 1],
                                        strides=[1, 1, 3, 1],
                                        padding='SAME',
                                        pool_func=tf.nn.max_pool,
                                        name='pooling4')
        layer['conv5'] = layers.conv(layer['pooling4'],
                                     filter=[1, 3, 32, 64],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv5')
        layer['pooling5'] = layers.pool(layer['conv5'],
                                        ksize=[1, 1, 3, 1],
                                        strides=[1, 1, 3, 1],
                                        padding='SAME',
                                        pool_func=tf.nn.max_pool,
                                        name='pooling5')
        layer['conv6'] = layers.conv(layer['pooling5'],
                                     filter=[1, 9, 64, 128],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv6')
        layer['conv7'] = layers.conv(layer['conv6'],
                                     filter=[1, 1, 128, 128],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv7')
        layer['conv8'] = layers.conv(layer['conv7'],
                                     filter=[1, 1, 128, 2],
                                     # strides=[1, 1, 4, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     wd=0.001,
                                     bias=0.0,
                                     name='conv8')


        layer['unfold'] = layers.unfold(layer['conv8'], name='unfold')
        #layer['logits'] = tf.reduce_mean(layer['unfold'], 1, name='logits')
        layer['class_prob'] = tf.nn.softmax(layer['unfold'], name='class_prob')
        layer['class_prediction'] = tf.argmax(layer['class_prob'], 1, name='class_pred')

        return layer

    def setup_loss(self):
        with tf.name_scope('loss'):
            raw_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.layer['unfold'],
                                                               labels=self.layer['target']))

            tf.summary.scalar('raw_loss', raw_loss)#tensorboard可视化
            tf.add_to_collection('losses', raw_loss)  #将raw_loss添加到losses中
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')#所有元素相加
            tf.summary.scalar('total_loss', loss) #tensorboard可视化

        return loss

    def setup_metrics(self):
        metrics = dict()
        with tf.variable_scope('metrics'):
            metrics['accuracy'] = tf.metrics.accuracy(labels=self.layer['target'],
                                                      predictions=self.layer['class_prediction'],
                                                      name='accuracy')[1]
            tf.summary.scalar('accuracy', metrics['accuracy'])
            metrics['recall'] = tf.metrics.recall(labels=self.layer['target'],
                                                  predictions=self.layer['class_prediction'],
                                                  name='recall')[1]
            tf.summary.scalar('recall', metrics['recall'])
            metrics['precision'] = tf.metrics.precision(labels=self.layer['target'],
                                                        predictions=self.layer['class_prediction'],
                                                        name='precision')[1]
            tf.summary.scalar('precision', metrics['precision'])
        return metrics

    def train(self, passes, new_training=True):
        with tf.Session() as sess:
            global_step=tf.Variable(0, trainable=False)
            #learning_rate = tf.train.exponential_decay(0.001, global_step, 200, 0.8, staircase=True)
            training = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

            if new_training:
                saver, global_step = Model.start_new_session(sess)
            else:
                saver, global_step = Model.continue_previous_session(sess,
                                                                     model_file='cnn',
                                                                     ckpt_file= self.config.root + '/event_detect/saver/cnn/checkpoint')

            sess.run(tf.local_variables_initializer())
            self.train_writer.add_graph(sess.graph, global_step=global_step)

            test_restlt=[]

            for step in range(1 + global_step, 1 + passes + global_step):
                input, target = self.reader.get_cnn_batch_data('train')
                #print(input.shape)
                summary, _, acc = sess.run([self.merged, training, self.metrics['accuracy']],
                                           feed_dict={self.layer['input']: input,
                                                      self.layer['target']: target})
                self.train_writer.add_summary(summary, step)

                if step % 10 == 0:
                    loss = sess.run(self.loss,
                                    feed_dict={self.layer['input']: input,
                                               self.layer['target']: target})
                    test_restlt.append(loss)

                    print("gobal_step {}, training_loss {}, accuracy {}".format(step, loss, acc))

                if step % 100 == 0:
                     test_x, text_y = self.reader.get_cnn_batch_data('test')
                     acc, recall, precision = sess.run([self.metrics['accuracy'],
                                                        self.metrics['recall'],
                                                        self.metrics['precision']],
                                                       feed_dict={self.layer['input']: test_x,
                                                                  self.layer['target']: text_y})


                     print("test: accuracy {}, recall {}, precision {}".format(acc, recall, precision))
                     saver.save(sess, self.config.root + '/event_detect/saver/cnn/cnn', global_step=step)
                     print('checkpoint saved')
                     #print(sess.run([self.layer['class_prob']], feed_dict={self.layer['input']: input}))

            print(test_restlt)

    def classify(self, sess, input_):
        class_prediction, confidence = sess.run([self.layer['class_prediction'], self.layer['class_prob']],
                                                feed_dict={self.layer['input']: input_})
        confidence = confidence[:, 1]
        if confidence > self.config.prob:
            class_prediction = 1
        else:
            class_prediction = 0
        return class_prediction, confidence

if __name__ == '__main__':
    cnn = CNN()
    cnn.train(Config().iteration, new_training=True)
