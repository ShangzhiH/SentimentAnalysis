# encoding=utf-8
import os
import time
import logging

import tensorflow as tf

from model_dataset import DatasetMaker
from data_utils import line_num_count
from utils import load_flags, print_flags
from model import EvalModel

flags = tf.app.flags
flags.DEFINE_string("root_path", "", "project root path")
flags.DEFINE_integer("batch_size", 512, "batch size")
flags.DEFINE_string("valid_data", "data/valid_ner", "validation data source")
flags.DEFINE_string("test_data", "data/test_ner", "test data source")

flags.DEFINE_string("best_model_path", "best_model/", "top N model")
flags.DEFINE_integer("N_best_model", 10, "models of top N accuracy")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))


class Eval(object):
    def __init__(self):
        self.root_path = FLAGS.root_path
        self.flag_config_file = os.path.join(self.root_path, "config.pkl")
        self.FLAGS = load_flags(self.flag_config_file, FLAGS, True)
        print_flags(self.FLAGS, True)

        self.valid_data = self.FLAGS.valid_data
        self.test_data = self.FLAGS.test_data

        self.valid_data_num = line_num_count(self.valid_data)
        self.test_data_num = line_num_count(self.test_data)
        tf.logging.info("{} / {} sentences in dev / test".format(self.valid_data_num, self.test_data_num))

        self.map_file = os.path.join(self.root_path, "map.pkl")
        self.vocabulary_file = os.path.join(self.root_path, "vocabulary.csv")
        self.eval_summary_op = None

        self.log_dir = os.path.join(self.root_path, self.FLAGS.log_dir)
        self.best_model_dir = os.path.join(self.root_path, self.FLAGS.log_dir, self.FLAGS.best_model_path)
        self.summary_dir = os.path.join(self.log_dir, "summary")
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)
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
                model.saver.save(session, os.path.join(self.best_model_dir, "model.ckpt"), self.worst_valid_model_index)
                tf.logging.info("Replacing model in {} by current model".format(
                    os.path.join(self.best_model_dir, "model.ckpt-") + str(self.worst_valid_model_index)))
                self.model_performance[self.worst_valid_model_index] = accuracy
                self.worst_valid_model_index = sorted(self.model_performance.items(), key=lambda d: d[1])[0][0]
            if accuracy > self.best_valid_accuracy:
                self.best_valid_accuracy = accuracy
                tf.logging.info("New best validation accuracy: {:.2f}%".format(100.0 * accuracy))
                model.saver.save(session, os.path.join(self.best_model_dir, "best_model.ckpt"))
                tf.logging.info("Saving best model in {}".format(os.path.join(self.best_model_dir, "best_model.ckpt")))
        elif name == "test":
            if accuracy > self.best_test_accuracy:
                self.best_test_accuracy = accuracy
                tf.logging.info("New best test accuracy: {:.2f}%".format(100.0 * accuracy))
        return accuracy

    def _init_dataset_maker(self):
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

    def eval(self):
        self._init_dataset_maker()

        eval_char_mapping_tensor, eval_label_mapping_tensor = DatasetMaker.make_mapping_table_tensor()
        valid_dataset = DatasetMaker.make_dataset(eval_char_mapping_tensor, eval_label_mapping_tensor, self.valid_data,
                                                  512, "eval", 1, 0)
        tf.logging.info("The part 1/1 Validation dataset is prepared!")
        test_dataset = DatasetMaker.make_dataset(eval_char_mapping_tensor, eval_label_mapping_tensor, self.test_data,
                                                 512, "eval", 1, 0)
        tf.logging.info("The part 1/1 Test dataset is prepared!")

        eval_iter = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
        valid_init_op = eval_iter.make_initializer(valid_dataset)
        test_init_op = eval_iter.make_initializer(test_dataset)
        eval_model = EvalModel(eval_iter, FLAGS)

        eval_session = self._create_session(None)
        eval_session.run(eval_char_mapping_tensor.init)
        eval_session.run(eval_label_mapping_tensor.init)
        last_step = -1
        while True:
            time.sleep(15)
            model_path = tf.train.latest_checkpoint(self.log_dir)
            if not model_path:
                tf.logging.info("model isn't found in log directory: waiting...")
                continue
            step = int(model_path.split("-")[-1])
            if step == last_step:
                tf.logging.info("model isn't updated(latest is step: {}) in log directory: waiting...".format(last_step))
                continue
            else:
                last_step = step

            tf.logging.info("Evaluate Validation Dataset and Test Dataset in step: {}".format(step))

            eval_model.saver.restore(eval_session, model_path)
            tf.logging.info("Loading model from {}".format(model_path))
            validation_accuracy = self._eval_performance(eval_session, eval_model, "validation", valid_init_op)
            test_accuracy = self._eval_performance(eval_session, eval_model, "test", test_init_op)
            eval_model.save_dev_test_summary(self.summary_writer, eval_session, validation_accuracy, test_accuracy, step)


def main(_):
    evaler = Eval()
    while True:
        if not tf.gfile.Exists(evaler.log_dir):
            tf.logging.error("Log directory doesn't exist!")
            time.sleep(10)
        else:
            break

    if not tf.gfile.Exists(evaler.best_model_dir):
        tf.gfile.MakeDirs(evaler.best_model_dir)
    if not tf.gfile.Exists(evaler.summary_dir):
        tf.gfile.MakeDirs(evaler.summary_dir)

    evaler.eval()

if __name__ == "__main__":
    tf.logging.info("----start----")
    tf.app.run()
