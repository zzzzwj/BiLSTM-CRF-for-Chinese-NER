import os, time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.epoch_num = kwargs["epoch"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.dropout_keep_prob = kwargs["dropout"]
        self.lr = kwargs["lr"]
        self.embeddings = kwargs["embeddings"]
        self.clip_grad = 5.0
        self.tag2label = kwargs["tag2label"]
        self.num_tags = len(self.tag2label)
        self.word2id = kwargs["word2id"]

        self.train_data = kwargs["train_data"]
        self.dev_data = kwargs["dev_data"]
        self.test_data = kwargs["test_data"]

        self.model_path = os.path.join(kwargs["outputdir"], 'checkpoints')
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.result_path = os.path.join(kwargs["outputdir"], 'tmp_results')
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        self.config = kwargs["config"]

        self.build_graph()

    def build_graph(self):
        # Add the graph place holder
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        # prepare the word embedding set
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=True, name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)

        # feed data into BiLSTM
        with tf.variable_scope("bilstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

            w = tf.get_variable(name="weight", shape=[2 * self.hidden_dim, self.num_tags], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            b = tf.get_variable(name="bias", shape=[self.num_tags], initializer=tf.zeros_initializer(), dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, w) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

        # calculate loss
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)

        # update the trainable parameter
        with tf.variable_scope("update_param"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars, global_step=self.global_step)

        # initialize all parameters
        self.init_op = tf.global_variables_initializer()

    def train(self):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, epoch)
                saver.save(sess, self.model_path + '/model', global_step = epoch)
            os.rmdir(self.result_path)

    def run_one_epoch(self, sess, epoch):
        num_batches = (len(self.train_data) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(self.train_data, self.batch_size, self.word2id, self.tag2label, shuffle=True)
        for step, (seqs, labels) in enumerate(batches):
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, _ = sess.run([self.train_op, self.loss, self.global_step],
                                                         feed_dict=feed_dict)
            print('\r{} batch:{}/{}, epoch:{}/{}, loss:{:.6}'.format(
                start_time, step + 1, num_batches, epoch + 1, self.epoch_num, loss_train), end='')
        print()
        if self.dev_data is not None:
            print()
            print('===========validation / test===========')
            label_list_dev, _ = self.dev_one_epoch(sess, self.dev_data)
            self.evaluate(label_list_dev, self.dev_data)
            print()

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, data):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(data, self.batch_size, self.word2id, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        logits, transition_params = sess.run([self.logits, self.transition_params],
                                             feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    def evaluate(self, label_list, data):
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        label_path = os.path.join(self.result_path, 'label')
        metric_path = os.path.join(self.result_path, 'result_metric')
        for _ in conlleval(model_predict, label_path, metric_path):
            print(_)
        os.remove(label_path)
        os.remove(metric_path)

    def predict(self, model_path):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, self.test_data)

        label2tag = {}
        for tag in self.tag2label.keys():
            label2tag[self.tag2label[tag]] = tag
        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                label_list[i][j] = label2tag[label_list[i][j]]
        return label_list