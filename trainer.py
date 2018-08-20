# encoding=utf-8
import os
import logging

import tensorflow as tf
import numpy as np

from model import TrainModel, EvalModel
from model_dataset import DatasetMaker
from data_utils import line_num_count
from utils import print_flags, save_flags

flags = tf.app.flags
flags.DEFINE_boolean("is_sync", False, "Whether use sync strategy to update parameter")

flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_boolean("use_attention", True, "Whether to use attention mechanism")
flags.DEFINE_float("l2_reg", 0.001, "l2 regularization value")
flags.DEFINE_float("clip", 5.0, "gradient clipper value")
flags.DEFINE_integer("max_epoch", 1000, "the max number of epochs")
flags.DEFINE_integer("batch_size", 2, "batch size")
flags.DEFINE_integer("check_step", 10, "Check loss every N steps")
flags.DEFINE_integer("eval_step", 50, "Eval model every N steps")
flags.DEFINE_string("root_path", "", "project root path")
flags.DEFINE_string("log_dir", "log/", "log directory")
flags.DEFINE_string("train_data", "data/example.train", "training data source")
flags.DEFINE_string("valid_data", "data/example.valid", "validation data source")
flags.DEFINE_string("test_data", "data/example.test", "test data source")
flags.DEFINE_integer("N_best_model", 10, "models of top N accuracy")

flags.DEFINE_integer("char_dim", 100, "char embedding dimension")
flags.DEFINE_string("rnn_type", "LSTM", "rnn cell type")
flags.DEFINE_integer("rnn_dim", 100, "rnn hidden dimension")
flags.DEFINE_integer("rnn_layer", 1, "rnn layer number")
flags.DEFINE_float("dropout", 0.5, "dropout rate during training")

flags.DEFINE_integer("worker_num", 0, "worker num to decide during trainer initialization")
flags.DEFINE_integer("char_num", 0, "char num to decide during trainer initialization")
flags.DEFINE_integer("label_num", 0, "label num to decide during trainer initialization")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))


class Trainer(object):
    def __init__(self):
        FLAGS.worker_num = 1
        self.root_path = FLAGS.root_path
        self.log_dir = os.path.join(self.root_path, FLAGS.log_dir)
        tf.logging.info("Project root path is: {}".format(self.root_path))
        tf.logging.info("Log directory is: {}".format(self.log_dir))

        self.train_data = FLAGS.train_data
        self.valid_data = FLAGS.valid_data
        self.test_data = FLAGS.test_data
        self.train_data_num = line_num_count(self.train_data)
        self.valid_data_num = line_num_count(self.valid_data)
        self.test_data_num = line_num_count(self.test_data)
        tf.logging.info("{} / {} / {} sentences in train /dev / test".format(self.train_data_num, self.valid_data_num, self.test_data_num))

        self.map_file = os.path.join(self.root_path, "map.pkl")
        self.vocabulary_file = os.path.join(self.root_path, "vocabulary.csv")

        self.global_step = 0
        self.check_step = FLAGS.check_step
        self.eval_step = FLAGS.eval_step
        self.train_summary_op = None
        self.eval_summary_op = None
        self.summary_writer = tf.summary.FileWriter(self.log_dir)
        self.topN = FLAGS.N_best_model
        self.model_performance = dict.fromkeys(range(self.topN), 0.0)
        self.worst_valid_model_index = 0
        self.best_valid_accuracy = 0.0
        self.best_test_accuracy = 0.0

    def _eval_performance(self, session, model, name, iter_init_op):
        tf.logging.info("Evaluate:{}".format(name))
        session.run(iter_init_op)
        tf.logging.info("Iterator is switched to {}".format(name))

        metric_dict = model.evaluate(session)
        all_real = sum([v["real"] for v in metric_dict.values()])
        all_correct = sum([v["correct"] for v in metric_dict.values()])
        all_predict = sum([v["predict"] for v in metric_dict.values()])
        accuracy = 1.0 * all_correct / all_predict
        tf.logging.info("Processed: {} sentences; correct: {}; accuracy {:.2f}%"
                        .format(all_real, all_correct, 100.0 * accuracy))

        for key, value in sorted(metric_dict.items(), key=lambda d: d[0]):
            precision = 0.0 if value["predict"] == 0 else 100.0 * value["correct"] / value["predict"]
            recall = 0.0 if value["real"] == 0 else 100.0 * value["correct"] / value["real"]
            f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
            tf.logging.info("Processed: {} '{}' labels: found: {}; correct: {};".
                            format(value["real"], key, value["predict"], value["correct"]))
            tf.logging.info(" ------------------------- precision: {:.2f}%; recall: {:.2f}%; f1: {:.2f}%"
                            .format(precision, recall, f1))
        if name == "validation":
            if accuracy > self.model_performance[self.worst_valid_model_index]:
                model.saver.save(session, os.path.join(self.log_dir, "model.ckpt"), self.worst_valid_model_index)
                tf.logging.info("Replacing model in {} by current model".format(
                    os.path.join(self.log_dir, "model.ckpt-") + str(self.worst_valid_model_index)))
                self.model_performance[self.worst_valid_model_index] = accuracy
                self.worst_valid_model_index = sorted(self.model_performance.items(), key=lambda d: d[1])[0][0]
            if accuracy > self.best_valid_accuracy:
                self.best_valid_accuracy = accuracy
                tf.logging.info("New best validation accuracy: {:.2f}%".format(100.0 * accuracy))
                model.saver.save(session, os.path.join(self.log_dir, "best_model.ckpt"))
                tf.logging.info("Saving best model in {}".format(os.path.join(self.log_dir, "best_model.ckpt")))
        elif name == "test":
            if accuracy > self.best_test_accuracy:
                self.best_test_accuracy = accuracy
                tf.logging.info("New best test accuracy: {:.2f}%".format(100.0 * accuracy))
        return accuracy

    def _init_dataset_maker(self, load=False):
        if not load:
            DatasetMaker.generate_mapping(self.train_data)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        else:
            DatasetMaker.load_mapping(self.map_file)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        FLAGS.char_num = len(DatasetMaker.char_to_id)
        FLAGS.label_num = len(DatasetMaker.label_to_id)

    @staticmethod
    def _create_session(graph):
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config, graph=graph)
        return session

    def train(self):
        self._init_dataset_maker(False)

        train_graph = tf.Graph()
        with train_graph.as_default():
            train_char_mapping_tensor, train_label_mapping_tensor = DatasetMaker.make_mapping_table_tensor()
            train_dataset = DatasetMaker.make_dataset(train_char_mapping_tensor, train_label_mapping_tensor, self.train_data, FLAGS.batch_size,
                                                      "train", 1, 0)
            self.global_step = tf.train.get_or_create_global_step()
            train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = train_iter.make_initializer(train_dataset)
            train_model = TrainModel(train_iter, FLAGS, self.global_step)
            self.train_summary_op = train_model.merge_train_summary_op

        eval_graph = tf.Graph()
        with eval_graph.as_default():
            eval_char_mapping_tensor, eval_label_mapping_tensor = DatasetMaker.make_mapping_table_tensor()
            valid_dataset = DatasetMaker.make_dataset(eval_char_mapping_tensor, eval_label_mapping_tensor, self.valid_data, FLAGS.batch_size, "eval", 1, 0)
            tf.logging.info("The part 1/1 Validation dataset is prepared!")
            test_dataset = DatasetMaker.make_dataset(eval_char_mapping_tensor, eval_label_mapping_tensor, self.test_data, FLAGS.batch_size, "eval", 1, 0)
            tf.logging.info("The part 1/1 Test dataset is prepared!")

            eval_iter = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
            valid_init_op = eval_iter.make_initializer(valid_dataset)
            test_init_op = eval_iter.make_initializer(test_dataset)
            eval_model = EvalModel(eval_iter, FLAGS)

        train_session = self._create_session(train_graph)
        tf.logging.info("Created model with fresh parameters.")
        print_flags(FLAGS)
        save_flags(FLAGS, os.path.join(self.root_path, "config.pkl"))
        with train_session.graph.as_default():
            train_session.run(tf.global_variables_initializer())
        train_session.run(train_char_mapping_tensor.init)
        train_session.run(train_label_mapping_tensor.init)
        train_session.run(train_init_op)

        eval_session = self._create_session(eval_graph)
        eval_session.run(eval_char_mapping_tensor.init)
        eval_session.run(eval_label_mapping_tensor.init)

        tf.logging.info("Start training")
        loss = []
        steps_per_epoch = self.train_data_num // FLAGS.batch_size  # how many batches in an epoch
        for i in range(FLAGS.max_epoch):
            for j in range(steps_per_epoch):
                step, loss_value = train_model.train(train_session)
                loss.append(loss_value)
                if step % FLAGS.check_step == 0:
                    iteration = step // steps_per_epoch + 1
                    tf.logging.info(
                        "iteration:{} step:{}/{}, cross entropy loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch,
                            steps_per_epoch, np.mean(loss)))
                    loss = []

                if step % FLAGS.eval_step == 0:
                    tf.logging.info("Evaluate Validation Dataset and Test Dataset in step: {}".format(step))
                    train_model.saver.save(train_session, os.path.join(self.log_dir, "temp_model.ckpt"))
                    tf.logging.info("Saving model parameters in {}".format(os.path.join(self.log_dir, "temp_model.ckpt")))

                    eval_model.saver.restore(eval_session, os.path.join(self.log_dir, "temp_model.ckpt"))
                    tf.logging.info("Loading model from {}".format(os.path.join(self.log_dir, "temp_model.ckpt")))
                    validation_accuracy = self._eval_performance(eval_session, eval_model, "validation", valid_init_op)
                    test_accuracy = self._eval_performance(eval_session, eval_model, "test", test_init_op)
                    eval_model.save_dev_test_summary(self.summary_writer, eval_session, validation_accuracy, test_accuracy, step)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    tf.logging.info("----start----")
    tf.app.run()
