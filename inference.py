# encoding=utf-8
import os
import logging

import numpy as np
import tensorflow as tf

from utils import print_flags
from data_utils import line_num_count
from model_dataset import DatasetMaker

flags = tf.app.flags
flags.DEFINE_string("root_path", "", "project root path")
flags.DEFINE_string("model_path", "data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression_tf", "model saved path")
flags.DEFINE_string("infer_data", "data/example.test", "inference data source")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))


class Inference(object):
    def __init__(self):
        self.root_path = FLAGS.root_path
        self.model_path = FLAGS.model_path
        self.flag_config_file = os.path.join(self.root_path, "config.pkl")
        self.FLAGS = FLAGS
        print_flags(self.FLAGS, False)

        self.infer_data = self.FLAGS.infer_data
        self.infer_data_num = line_num_count(self.infer_data)
        tf.logging.info("{} sentences in infer".format(self.infer_data_num))

        self.map_file = os.path.join(self.root_path, "data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression_word_pos_dict.pkl")
        self.vocabulary_file = os.path.join(self.root_path, "vocabulary.csv")

        self.train_init_op = None

    def _init_dataset_maker(self):
        DatasetMaker.load_mapping(self.map_file)
        # DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        FLAGS.char_num = len(DatasetMaker.char_to_id)
        FLAGS.label_num = len(DatasetMaker.label_to_id)

    @staticmethod
    def _create_session(graph):
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config, graph=graph)
        return session

    def infer(self):
        self._init_dataset_maker()

        char_mapping_tensor, label_mapping_tensor = DatasetMaker.make_mapping_table_tensor()
        infer_dataset = DatasetMaker.make_dataset(char_mapping_tensor, label_mapping_tensor, self.infer_data,
                                                  2, "infer", 1, 0)
        tf.logging.info("The part {}/{} Training dataset is prepared!".format(1, 1))
        train_iter = tf.data.Iterator.from_structure(infer_dataset.output_types, infer_dataset.output_shapes)
        self.train_init_op = train_iter.make_initializer(infer_dataset)

        infer_session = self._create_session(None)
        infer_session.run(char_mapping_tensor.init)
        infer_session.run(self.train_init_op)

        tf.saved_model.loader.load(infer_session, ["sentiment-analysis"], self.model_path)
        graph = tf.get_default_graph()
        x_origin = graph.get_tensor_by_name("input_1:0")
        y = graph.get_tensor_by_name("dense_3/Sigmoid:0")

        x = train_iter.get_next()
        xx = infer_session.run(x)
        xx = [line[::-1] for line in xx]
        print(xx)
        s = [1268, 7,468,1,428,85,44,331,76,2,60,354,2,8,68,221,2,4281,270,89,667,748,249]
        print(infer_session.run(y, {x_origin:xx}))
        tf.logging.info("Loading model from {}".format(self.model_path))
        """with tf.gfile.GFile("file_{}".format(self.task_index), "w") as f_w:
            try:
                infer_model.infer(infer_session, f_w)
            except tf.errors.OutOfRangeError:
                tf.logging.info("Data in worker {} is finished!".format(self.task_index))"""


def main(_):
    infer = Inference()
    infer.infer()

if __name__ == "__main__":
    tf.logging.info("----start----")
    tf.app.run()
