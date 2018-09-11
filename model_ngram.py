# encoding=utf-8
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell

from model_dataset_ngram import DatasetMaker
from eval_utils import metric_collect

__all__ = ["TrainModel", "EvalModel", "InferModel"]


class BaseModel(object):
    def __init__(self, input_chars, input_2grams, input_3grams, flags, dropout):
        self.char_dim = flags.char_dim
        self.char_num = flags.char_num
        self.gram2_num = flags.gram2_num
        self.gram3_num = flags.gram3_num
        self.label_num = flags.label_num

        self.rnn_type = flags.rnn_type
        self.rnn_dim = flags.rnn_dim
        self.rnn_layer = flags.rnn_layer

        self.dropout = dropout
        self.initializer = xavier_initializer()
        self.use_attention = flags.use_attention
        self.l2_reg = flags.l2_reg

        self.input_chars = input_chars
        self.input_2grams = input_2grams
        self.input_3grams = input_3grams
        real_char = tf.sign(self.input_chars)
        self.char_len = tf.reduce_sum(real_char, reduction_indices=1)
        self.gram2_len = self.char_len - 1
        self.gram3_len = self.char_len - 2
        if self.use_attention:
            self.weights = {
                    "attention_Wm": tf.get_variable(name="attention_Wm",
                                                    shape=[6 * self.rnn_dim, 1],
                                                    initializer=self.initializer,
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                                                    ),

                    "attention_Wr": tf.get_variable(name="attention_Wr",
                                                    shape=[6 * self.rnn_dim, self.rnn_dim],
                                                    initializer=self.initializer,
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                                                    ),

                    "attention_Wn": tf.get_variable(name="attention_Wn",
                                                    shape=[6 * self.rnn_dim, self.rnn_dim],
                                                    initializer=self.initializer,
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                                                    )
            }

    def build_graph(self):
        # embedding
        input_embedding = self._build_embedding_layer(self.input_chars)
        # rnn
        self.rnn_output = self._build_multilayer_rnn(input_embedding, self.rnn_type, self.rnn_dim, self.rnn_layer, self.char_len, name="Char")
        # 2gram embedding
        gram2_embedding = self._build_gram2_embedding_layer(self.input_2grams)
        self.rnn_output_gram2 = self._build_multilayer_rnn(gram2_embedding, self.rnn_type, self.rnn_dim, self.rnn_layer, self.gram2_len, name="Gram2")
        # 3gram embedding
        gram3_embedding = self._build_gram3_embedding_layer(self.input_3grams)
        self.rnn_output_gram3 = self._build_multilayer_rnn(gram3_embedding, self.rnn_type, self.rnn_dim, self.rnn_layer, self.gram3_len, name="Gram3")

        # concat final step output of biRNN
        final_rnn_output_char = self._pick_last_output(self.rnn_output, length=self.char_len)
        final_rnn_output_gram2 = self._pick_last_output(self.rnn_output_gram2, length=self.gram2_len)
        final_rnn_output_gram3 = self._pick_last_output(self.rnn_output_gram3, length=self.gram3_len)
        final_rnn_output = tf.concat([final_rnn_output_char, final_rnn_output_gram2, final_rnn_output_gram3], axis=1)


        # projection
        logits = self._build_projection_layer(final_rnn_output, self.label_num)
        return logits

    @staticmethod
    def _softmax_for_attention(inputs, length):
        """
        Args:
            inputs: input to generate attention weight matrix, shape = [batch_size, 1, max_len]
            length: real length of input sentences, shape = [batch_size]
        Returns:
            attention weight matrix, shape=[batch_size, 1, max_len]
        """
        max_len = tf.shape(inputs)[2]
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def _build_embedding_layer(self, inputs):
        with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name="char_embedding_lookup_table", shape=[self.char_num, self.char_dim], partitioner=tf.fixed_size_partitioner(6))
        return tf.nn.dropout(tf.nn.embedding_lookup(self.char_lookup, inputs), keep_prob=self.dropout)

    def _build_gram2_embedding_layer(self, inputs):
        with tf.variable_scope("gram2_embedding", reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
            self.gram2_lookup = tf.get_variable(name="gram2_embedding_lookup_table", shape=[self.gram2_num, self.char_dim], partitioner=tf.fixed_size_partitioner(6))
        return tf.nn.dropout(tf.nn.embedding_lookup(self.gram2_lookup, inputs), keep_prob=self.dropout)

    def _build_gram3_embedding_layer(self, inputs):
        with tf.variable_scope("gram3_embedding", reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
            self.gram3_lookup = tf.get_variable(name="gram3_embedding_lookup_table", shape=[self.gram3_num, self.char_dim], partitioner=tf.fixed_size_partitioner(6))
        return tf.nn.dropout(tf.nn.embedding_lookup(self.gram3_lookup, inputs), keep_prob=self.dropout)


    def _create_rnn_cell(self, rnn_type, rnn_dim, rnn_layer):
        def _single_rnn_cell():
            single_cell = None
            if rnn_type == "LSTM":
                single_cell = LSTMCell(rnn_dim, initializer=self.initializer, use_peepholes=True)
            elif rnn_type == "GRU":
                single_cell = GRUCell(rnn_dim, kernel_initializer=self.initializer)
            cell = DropoutWrapper(single_cell, output_keep_prob=self.dropout)
            return cell
        multi_cell = MultiRNNCell([_single_rnn_cell() for _ in range(rnn_layer)])
        return multi_cell

    def _build_birnn_layer(self, rnn_input, rnn_type, rnn_dim, lengths):
        with tf.variable_scope("forward_rnn", reuse=tf.AUTO_REUSE), tf.device("/gpu:0"):
            forward_rnn_cell = self._create_rnn_cell(rnn_type, rnn_dim, 1)
        with tf.variable_scope("backward_rnn", reuse=tf.AUTO_REUSE), tf.device("/gpu:1"):
            backward_rnn_cell = self._create_rnn_cell(rnn_type, rnn_dim, 1)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_rnn_cell, backward_rnn_cell, rnn_input, dtype=tf.float32,
                                                     sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def _build_multilayer_rnn(self, rnn_input, rnn_type, rnn_dim, rnn_layer, lengths, name=""):
        inputs = rnn_input
        for i in range(rnn_layer):
            with tf.variable_scope("{}_Bi{}_Sequence_Layer_{}".format(name, rnn_type, i+1), reuse=tf.AUTO_REUSE):
                inputs = self._build_birnn_layer(inputs, rnn_type, rnn_dim, lengths)
        return inputs

    @staticmethod
    def _extract_axis_1(data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def _pick_last_output(self, rnn_output, length):
        forward_output = self._extract_axis_1(rnn_output, length - 1)[:, :self.rnn_dim]
        backward_output = rnn_output[:, 0, self.rnn_dim:]
        assert forward_output.get_shape()[-1] == backward_output.get_shape()[-1]
        return tf.concat([forward_output, backward_output], axis=1)

    def _build_projection_layer(self, inputs, output_dim):
        with tf.variable_scope("projection_layer_1", reuse=tf.AUTO_REUSE):
            activated_inputs = tf.nn.selu(inputs)
            activated_bn_inputs = tf.layers.batch_normalization(activated_inputs)
            self.projection_layer_1 = tf.layers.Dense(units=6*self.rnn_dim, use_bias=True,
                                                      activation=tf.nn.selu, kernel_initializer=self.initializer,
                                                      name="projection_layer_1_dense",
                                                      activity_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.projection_output_1 = self.projection_layer_1.apply(activated_bn_inputs)
        with tf.variable_scope("final_projection_layer", reuse=tf.AUTO_REUSE):
            bn_projection_output_1 = tf.layers.batch_normalization(self.projection_output_1)
            self.projection_layer_final = tf.layers.Dense(units=output_dim, use_bias=True, kernel_initializer=self.initializer,
                                                name="final_projection_layer_dense",
                                                activity_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            logits = self.projection_layer_final.apply(bn_projection_output_1)
        return logits


class TrainModel(BaseModel):
    def __init__(self, iterator, flags, global_step):
        chars, grams2, grams3, labels = iterator.get_next()
        super(TrainModel, self).__init__(chars, grams2, grams3, flags, flags.dropout)
        self.lr = flags.lr
        self.clip = flags.clip
        self.is_sync = flags.is_sync
        self.worker_num = flags.worker_num
        self.global_step = global_step

        self.train_summary = []
        self.logits = self.build_graph()
        self.loss = self._build_loss_layer(self.logits, labels)
        self.train_op, self.optimizer = self._optimizer(self.loss, global_step)
        self.saver = tf.train.Saver(var_list=tf.global_variables(), sharded=True)
        self.merge_train_summary_op = tf.summary.merge(self.train_summary)

    def _build_loss_layer(self, inputs, labels):
        loss = self._softmax_cross_entropy_loss(inputs, labels)
        self.train_summary.append(tf.summary.scalar("training_loss", loss))
        return loss

    def _softmax_cross_entropy_loss(self, project_logits, labels):
        with tf.variable_scope("softmax_cross_entropy_loss"):
            # use weights to mask loss of padding position
            log_likelihood = tf.losses.sparse_softmax_cross_entropy(
                logits=project_logits, labels=labels
            )
            loss = tf.reduce_mean(log_likelihood)
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss += sum(reg_loss)
        return loss

    def _make_train_summary(self, grads_vars):
        # visualize gradient
        with tf.variable_scope("variable_gradient"):
            for g, v in grads_vars:
                if 'char_embedding_lookup_table' in v.name:
                    gradients = tf.sqrt(tf.reduce_mean(g.values ** 2))
                    self.train_summary.append(tf.summary.scalar("char_embedding_matrix_grad_norm", gradients))
                elif 'gram2_embedding_lookup_table' in v.name:
                    gradients = tf.sqrt(tf.reduce_mean(g.values ** 2))
                    self.train_summary.append(tf.summary.scalar("gram2_embedding_matrix_grad_norm", gradients))
                elif 'gram3_embedding_lookup_table' in v.name:
                    gradients = tf.sqrt(tf.reduce_mean(g.values ** 2))
                    self.train_summary.append(tf.summary.scalar("gram3_embedding_matrix_grad_norm", gradients))
                elif 'projection_layer_1_dense' in v.name:
                    if "kernel" in v.name:
                        gradients = tf.sqrt(tf.reduce_mean(g ** 2))
                        self.train_summary.append(tf.summary.scalar("projection_layer_1_W_grad_norm", gradients))
                    elif "bias" in v.name:
                        gradients = tf.norm(g)
                        self.train_summary.append(tf.summary.scalar("projection_layer_1_b_grad_norm", gradients))
                elif 'final_projection_layer_dense' in v.name:
                    if "kernel" in v.name:
                        gradients = tf.sqrt(tf.reduce_mean(g ** 2))
                        self.train_summary.append(tf.summary.scalar("projection_layer_final_W_grad_norm", gradients))
                    elif "bias" in v.name:
                        gradients = tf.norm(g)
                        self.train_summary.append(tf.summary.scalar("projection_layer_final_b_grad_norm", gradients))
        # viszalize weight
        with tf.variable_scope("variable_hist"):
            self.train_summary.append(tf.summary.histogram("char_embedding_matrix_hist", tf.reshape(self.char_lookup, [-1])))
            self.train_summary.append(tf.summary.histogram("gram2_embedding_matrix_hist", tf.reshape(self.gram2_lookup, [-1])))
            self.train_summary.append(tf.summary.histogram("gram3_embedding_matrix_hist", tf.reshape(self.gram3_lookup, [-1])))
            self.train_summary.append(tf.summary.histogram("projection_layer_1_w_hist", tf.reshape(self.projection_layer_1.weights[0], [-1])))
            self.train_summary.append(tf.summary.histogram("projection_layer_1_b_hist", tf.reshape(self.projection_layer_1.weights[1], [-1])))
            self.train_summary.append(tf.summary.histogram("projection_layer_final_w_hist", tf.reshape(self.projection_layer_final.weights[0], [-1])))
            self.train_summary.append(tf.summary.histogram("projection_layer_final_b_hist", tf.reshape(self.projection_layer_final.weights[1], [-1])))

        # visualize rnn output
        with tf.variable_scope("sample_rnn_output_different_step"):
            self.train_summary.append(tf.summary.histogram("sample_1_rnn_output", tf.norm(self.rnn_output[0], axis=1)))
            self.train_summary.append(tf.summary.histogram("sample_2_rnn_output", tf.norm(self.rnn_output[1], axis=1)))
            self.train_summary.append(tf.summary.histogram("sample_3_rnn_output", tf.norm(self.rnn_output[2], axis=1)))
            self.train_summary.append(tf.summary.histogram("sample_4_rnn_output", tf.norm(self.rnn_output[3], axis=1)))
        # visualize projection output
        with tf.variable_scope("sample_projection_1_output"):
            self.train_summary.append(tf.summary.histogram("sample_1_projection_1_output", self.projection_output_1[0]))
            self.train_summary.append(tf.summary.histogram("sample_2_projection_1_output", self.projection_output_1[1]))
            self.train_summary.append(tf.summary.histogram("sample_3_projection_1_output", self.projection_output_1[2]))
            self.train_summary.append(tf.summary.histogram("sample_4_projection_1_output", self.projection_output_1[3]))
        with tf.variable_scope("sample_projection_final_output"):
            self.train_summary.append(tf.summary.histogram("sample_1_projection_final_output", self.logits[0]))
            self.train_summary.append(tf.summary.histogram("sample_2_projection_final_output", self.logits[1]))
            self.train_summary.append(tf.summary.histogram("sample_3_projection_final_output", self.logits[2]))
            self.train_summary.append(tf.summary.histogram("sample_4_projection_final_output", self.logits[3]))

    def _optimizer(self, loss, global_step):
        optimizer = tf.train.AdamOptimizer(self.lr)
        if self.is_sync:
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=self.worker_num,
                                                       total_num_replicas=self.worker_num)
        grads_vars = optimizer.compute_gradients(loss)
        self._make_train_summary(grads_vars)
        capped_grads_vars = [[tf.clip_by_norm(g, self.clip), v] for g, v in grads_vars]
        train_op = optimizer.apply_gradients(capped_grads_vars, global_step)
        return train_op, optimizer

    def train(self, session):
        step, loss_value, _ = session.run([self.global_step, self.loss, self.train_op])
        return step, loss_value


class EvalModel(BaseModel):
    def __init__(self, iterator, flags):
        self._make_eval_summary()
        chars, grams2, grams3, self.labels = iterator.get_next()

        super(EvalModel, self).__init__(chars, grams2, grams3, flags, 1.0)
        self.logits = self.build_graph()

        self.saver = tf.train.Saver(var_list=tf.global_variables(), sharded=True)

    @staticmethod
    def _logits_to_label_ids(logits):
        predict_label_ids = np.argmax(logits, axis=1)
        return predict_label_ids

    def evaluate(self, session):
        metric_dict = {}
        try:
            while True:
                real_label_ids, logits = session.run([self.labels, self.logits])
                predict_label_ids = self._logits_to_label_ids(logits)

                predict_labels = DatasetMaker.label_ids_to_labels(predict_label_ids)
                real_labels = DatasetMaker.label_ids_to_labels(real_label_ids)
                metric_dict = metric_collect(real_labels, predict_labels, metric_dict)
        except tf.errors.OutOfRangeError:
            return metric_dict

    def _make_eval_summary(self):
        self.eval_summary = []
        self.valid_accuracy = tf.placeholder(tf.float32, shape=None)
        self.test_accuracy = tf.placeholder(tf.float32, shape=None)
        with tf.variable_scope("performance"):
            self.eval_summary.append(tf.summary.scalar("valid_accuracy", self.valid_accuracy))
            self.eval_summary.append(tf.summary.scalar("test_accuracy", self.test_accuracy))
        self.merge_eval_summary_op = tf.summary.merge(self.eval_summary)

    def save_dev_test_summary(self, summary_writer, session, dev_accuracy, test_accuracy, step):
        merged_summary = session.run(self.merge_eval_summary_op, feed_dict={self.valid_accuracy: dev_accuracy,
                                                                            self.test_accuracy: test_accuracy})
        summary_writer.add_graph(session.graph)
        summary_writer.add_summary(merged_summary, step)


class InferModel(BaseModel):
    def __init__(self, iterator, flags):
        # id to record data while inference
        self.ids, chars = iterator.get_next()

        super(InferModel, self).__init__(chars, flags, 1.0)
        self.logits = self.build_graph()

    @staticmethod
    def _logits_to_label_ids(logits):
        predict_label_ids = np.argmax(logits, axis=1)
        return predict_label_ids

    def infer(self, session, file_handler):
        try:
            while True:
                data_ids, logits = session.run([self.ids, self.logits])
                predict_label_ids = self._logits_to_label_ids(logits)

                predict_labels = DatasetMaker.label_ids_to_labels(predict_label_ids)
                file_handler.write(np.concatenate([data_ids, predict_labels], axis=1))
        except tf.errors.OutOfRangeError as e:
            raise e



