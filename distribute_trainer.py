# encoding=utf-8
import os
import sys
import traceback
import time
import logging

import tensorflow as tf

from model import TrainModel
from model_dataset import DatasetMaker
from data_utils import line_num_count
from utils import print_flags, save_flags, IteratorInitializerHook, LoggingCheckpointSaverListener

flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of ps hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of worker hostname:port pairs")
flags.DEFINE_string("chief_hosts", "", "Comma-separated list of chief hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker' or 'chief'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_boolean("is_sync", True, "Whether use sync strategy to update parameter")

flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_boolean("use_attention", False, "Whether to use attention mechanism")
flags.DEFINE_float("l2_reg", 0.001, "l2 regularization value")
flags.DEFINE_float("clip", 5.0, "gradient clipper value")
flags.DEFINE_float("max_epoch", 1000, "the max number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("check_step", 100, "Check loss every N steps")
flags.DEFINE_integer("eval_step", 1000, "Eval model every N steps")
flags.DEFINE_string("root_path", "", "project root path")
flags.DEFINE_string("log_dir", "log/", "log directory")
flags.DEFINE_string("train_data", "data/train_ner", "training data source")

flags.DEFINE_integer("char_dim", 300, "char embedding dimension")
flags.DEFINE_string("rnn_type", "LSTM", "rnn cell type")
flags.DEFINE_integer("rnn_dim", 300, "rnn hidden dimension")
flags.DEFINE_integer("rnn_layer", 1, "rnn layer number")
flags.DEFINE_float("dropout", 0.5, "dropout rate during training")

flags.DEFINE_integer("worker_num", 0, "worker num to decide during trainer initialization")
flags.DEFINE_integer("char_num", 0, "char num to decide during trainer initialization")
flags.DEFINE_integer("label_num", 0, "label num to decide during trainer initialization")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


class Trainer(object):
    def __init__(self):
        self.ps_hosts = FLAGS.ps_hosts.split(",")
        self.worker_hosts = FLAGS.worker_hosts.split(",")
        self.chief_hosts = FLAGS.chief_hosts.split(",")
        tf.logging.info("PS hosts are: {}".format(self.ps_hosts))
        tf.logging.info("Worker hosts are: {}".format(self.worker_hosts))
        tf.logging.info("Chief hosts are: {}".format(self.chief_hosts))

        self.root_path = FLAGS.root_path
        self.log_dir = os.path.join(self.root_path, FLAGS.log_dir)
        tf.logging.info("Project root path is: {}".format(self.root_path))
        tf.logging.info("Log directory is: {}".format(self.log_dir))

        self.train_data = FLAGS.train_data
        self.train_data_num = line_num_count(self.train_data)
        tf.logging.info("{} sentences in train".format(self.train_data_num))
        self.num_steps = int(FLAGS.max_epoch * self.train_data_num / FLAGS.batch_size)

        self.map_file = os.path.join(self.root_path, "map.pkl")
        self.vocabulary_file = os.path.join(self.root_path, "vocabulary.csv")

        self.job_name = FLAGS.job_name
        self.task_index = FLAGS.task_index
        self.cluster = tf.train.ClusterSpec(
            {"ps": self.ps_hosts, "worker": self.worker_hosts, "chief": self.chief_hosts}
        )
        self.is_sync = FLAGS.is_sync
        self.server = tf.train.Server(self.cluster, job_name=self.job_name, task_index=self.task_index)
        self.is_chief = (self.job_name == "chief" and self.task_index == 0)
        self.worker_prefix = '/job:%s/task:%s' % (self.job_name, self.task_index)
        self.num_ps = self.cluster.num_tasks("ps")
        self.num_worker = self.cluster.num_tasks("worker")
        FLAGS.worker_num = self.num_worker

        self.global_step = 0
        self.check_step = FLAGS.check_step
        self.eval_step = FLAGS.eval_step
        self.train_summary_op = None
        self.train_init_op = None
        self.summary_writer = tf.summary.FileWriter(self.log_dir)

        self.session = None
        self.optimizer = None

    def _init_dataset_maker(self, load=False):
        if not load:
            DatasetMaker.generate_mapping(self.train_data)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        else:
            DatasetMaker.load_mapping(self.map_file)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        FLAGS.char_num = len(DatasetMaker.char_to_id)
        FLAGS.label_num = len(DatasetMaker.label_to_id)

    def _create_session(self):
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        device_filters=["/job:ps", "/job:{}/task:{}".format(self.job_name, self.task_index)],
                                        log_device_placement=False)
        session_config.gpu_options.allow_growth = True

        hooks = []
        if not self.is_chief:
            hooks.append(tf.train.StopAtStepHook(num_steps=self.num_steps))
            hooks.append(IteratorInitializerHook(self.train_init_op))
            # print first worker's loss summary
            if self.task_index == 0:
                hooks.append(tf.train.SummarySaverHook(save_steps=100, output_dir=self.log_dir, summary_op=self.train_summary_op))
        else:
            listener = LoggingCheckpointSaverListener()
            hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=self.log_dir, save_steps=self.eval_step, listeners=[listener]))

        if self.is_sync:
            hooks.append(self.optimizer.make_session_run_hook(self.is_chief))

        self.session = tf.train.MonitoredTrainingSession(
            master=self.server.target,
            is_chief=self.is_chief,
            checkpoint_dir=None,
            scaffold=None,
            hooks=hooks,
            chief_only_hooks=None,
            save_checkpoint_secs=None,
            save_summaries_secs=None,
            save_summaries_steps=None,
            config=session_config,
            stop_grace_period_secs=120
        )
        return self.session

    def _create_session_wrapper(self, retries=10):
        if retries == 0:
            tf.logging.error("Creating the session is out of times!")
            sys.exit(0)
        try:
            return self._create_session()
        except Exception as e:
            tf.logging.info(e)
            tf.logging.info("Retry creating session:{}".format(retries))
            try:
                if self.session is not None:
                    self.session.close()
                else:
                    tf.logging.info("Close session: session is None!")
            except Exception as e:
                exc_info = traceback.format_exc(sys.exc_info())
                msg = "Creating session exception:{}\n{}".format(e, exc_info)
                tf.logging.warn(msg)
            return self._create_session_wrapper(retries - 1)

    def train(self):
        if self.job_name == "ps":
            with tf.device("/cpu:0"):
                self.server.join()
                return

        self._init_dataset_maker(False)

        with tf.device(tf.train.replica_device_setter(worker_device=self.worker_prefix, cluster=self.cluster)):
            self.global_step = tf.train.get_or_create_global_step()
            char_mapping_tensor, label_mapping_tensor = DatasetMaker.make_mapping_table_tensor()

            train_dataset = DatasetMaker.make_dataset(char_mapping_tensor, label_mapping_tensor, self.train_data, FLAGS.batch_size,
                                                          "train", 1, 0)
            tf.logging.info("The part {}/{} Training dataset is prepared!".format(1, 1))
            train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            self.train_init_op = train_iter.make_initializer(train_dataset)

            train_model = TrainModel(train_iter, FLAGS, self.global_step)
            self.optimizer = train_model.optimizer
            self.train_summary_op = train_model.merge_train_summary_op

        with self._create_session_wrapper(retries=10) as sess:
            try:
                if self.job_name == "worker":
                    step = 0
                    while not sess.should_stop():
                        global_step_val, loss_value = train_model.train(sess)
                        if (step + 1) % self.check_step == 0:
                            epoch = ((step + 1) * FLAGS.batch_size) // self.train_data_num
                            tf.logging.info("Job-{}:Worker-{}-----Local_Step/Global_Step:{}/{}:Loss is {:.4f}".
                                            format(self.job_name, self.task_index, step, global_step_val, loss_value))
                            tf.logging.info("Epoch:{}-Processed {}/{} data".format(
                                epoch, (step + 1) * FLAGS.batch_size % self.train_data_num, self.train_data_num))
                        step += 1
                elif self.job_name == "chief":
                    print_flags(FLAGS, True)
                    save_flags(FLAGS, os.path.join(self.root_path, "config.pkl"), True)
                    tf.logging.info("Waiting for training...")
                    # record top N model's performance
                    while True:
                        time.sleep(2)
                        global_step_val = sess.run(self.global_step)
                        tf.logging.info("Global step is {}".format(global_step_val))
            except tf.errors.OutOfRangeError as e:
                exc_info = traceback.format_exc(sys.exc_info())
                msg = 'Out of range error:{}\n{}'.format(e, exc_info)
                tf.logging.warn(msg)
                tf.logging.info('Done training -- step limit reached')


def main(_):
    if FLAGS.job_name == "chief" and FLAGS.task_index == 0:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    trainer = Trainer()
    try:
        trainer.train()
    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:{}\n{}'.format(e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            tf.logging.warn(msg)
        else:
            tf.logging.error(msg)
        # 0 means 'be over', 1 means 'will retry'
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)

if __name__ == "__main__":
    tf.logging.info("----start----")
    tf.app.run()
