# encoding=utf-8
import sys
import pickle

import tensorflow as tf

from utils import unicode_open

__all__ = ["DatasetMaker"]


# full_to_half, replace_html.etc operations were done during preprocessing process to avoid redundant calculation
def _generator_maker(file_path, infer=False, return_unicode=True):
    """
    :rtype: generator
    """
    def _generator():
        for line in unicode_open(file_path):
            tokens = line.strip().split("\t")
            if not infer:
                if len(tokens) != 2:
                    continue
                sentence, label = tokens
                chars = list(sentence.strip().replace(u"\\\\r\\\\n", u"\n").replace(u"\\\\n", u"\n"))
                # 这里有个迷之bug, dataset.from_generator里的generator返回了unicode类型的话会报错
                # 因为指定类型是str的, 但是tensorflow的type又没有dtype=tf.UNICODE
                # 但是之前几个项目同样的做法又没有问题，纠结了3个小时候后我决定放弃，硬转成str吧，虽然如果打印会发现str表示中文是有问题的
                # 但是反正模型训练的时候都是转成id输入进embedding层的，只要保证一字一id就行了，至于字对程序来说具体长什么样无关紧要
                # ps: 升级tensorflow版本听说可以解决，但是早日摆脱python2才是正道
                if not return_unicode:
                    chars = [char.encode("utf-8") for char in chars]
                yield chars, label
            else:
                if len(tokens) != 1:
                    continue
                chars = list(tokens[0].strip().replace("\\\\r\\\\n", "\n").replace("\\\\n", "\n"))
                yield chars
    return _generator


class DatasetMaker(object):

    char_to_id = {u"<PAD>": 0, u"<UNK>": 1, u"<START>": 2, u"<END>": 3}
    id_to_char = {0: u"<PAD>", 1: u"<UNK>", 2: u"<START>", 3: u"<END>"}
    label_to_id = {}
    id_to_label = {}
    mapping_dict_ready = False

    @classmethod
    def label_ids_to_labels(cls, label_ids):
        if not cls.mapping_dict_ready:
            tf.logging.error("Mapping dict isn't initialized!")
            sys.exit(0)
        labels = [cls.id_to_label[label_id] for label_id in label_ids]
        return labels

    @classmethod
    def generate_mapping(cls, file_path):
        char_freq = {}
        for char_list, label in _generator_maker(file_path)():
            for char in char_list:
                char_freq[char] = char_freq.get(char, 0) + 1

            cls.label_to_id[label] = cls.label_to_id.get(label, len(cls.label_to_id))
            cls.id_to_label[cls.label_to_id[label]] = label

        sorted_items = sorted(char_freq.items(), key=lambda d: d[1], reverse=True)
        for key, value in sorted_items:
            if key not in cls.char_to_id:
                cls.char_to_id[key], cls.id_to_char[len(cls.id_to_char)] = len(cls.char_to_id), key

        cls.mapping_dict_ready = True
        tf.logging.info("Generated mapping dictionary with {} different chars and {} different labels!".
                        format(len(cls.char_to_id), len(cls.label_to_id)))

    @classmethod
    def save_mapping(cls, mapfile_path, vocabfile_path):
        if not cls.mapping_dict_ready:
            tf.logging.error("Error: mapping dict isn't initialized!")
            sys.exit(0)

        with tf.gfile.GFile(mapfile_path, "wb") as f_w:
            pickle.dump([cls.char_to_id, cls.id_to_char, cls.label_to_id, cls.id_to_label], f_w)
        tf.logging.info("Saved mapping dictionary in file {}".format(mapfile_path))
        with tf.gfile.GFile(vocabfile_path, "w") as f:
            f.write(u"\n".join(cls.char_to_id.keys()))
        tf.logging.info("Saved readable vocabulary in file {}".format(vocabfile_path))

    @classmethod
    def load_mapping(cls, mapfile_path):
        with tf.gfile.GFile(mapfile_path, "rb") as f:
            cls.char_to_id, cls.id_to_char, cls.label_to_id, cls.id_to_label = pickle.load(f)

        cls.mapping_dict_ready = True
        tf.logging.info("Loaded mapping dictionary from file {} with {} different chars and {} different labels!".
                        format(mapfile_path, len(cls.char_to_id), len(cls.label_to_id)))

    @classmethod
    def make_mapping_table_tensor(cls, name="mappings"):
        if not cls.mapping_dict_ready:
            tf.logging.error("Error: mapping dict isn't initialized!")
            sys.exit(0)

        char_mapping_tensor = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(cls.char_to_id.keys(), cls.char_to_id.values()),
            cls.char_to_id.get(u"<UNK>"), name=name
        )
        label_mapping_tensor = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(cls.label_to_id.keys(), cls.label_to_id.values()),
            0, name=name
        )
        tf.logging.info("Created mapping table tensor from exist mapping dict!")
        return char_mapping_tensor, label_mapping_tensor

    @staticmethod
    def make_dataset(char_mapping_tensor, label_mapping_tensor, file_path, batch_size, task_type, num_shards, worker_index):
        if task_type == "infer":
            dataset = tf.data.Dataset.from_generator(_generator_maker(file_path, True, False), tf.string, tf.TensorShape([None]))
            dataset = dataset.shard(num_shards, worker_index)
            dataset = dataset.map(lambda chars: (char_mapping_tensor.lookup(chars)))
            dataset = dataset.padded_batch(batch_size, padded_shapes=tf.TensorShape([None]))
        else:
            dataset = tf.data.Dataset.from_generator(_generator_maker(file_path, False, False), (tf.string, tf.string),
                                                     (tf.TensorShape([None]), tf.TensorShape([])))
            dataset = dataset.shard(num_shards, worker_index)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.map(lambda chars, label:
                                  (char_mapping_tensor.lookup(chars), label_mapping_tensor.lookup(label)))
            # train
            if task_type == "train":
                dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]),
                                                                          tf.TensorShape([]))).repeat()
            # eval
            elif task_type == "eval":
                dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]),
                                                                          tf.TensorShape([])))
        return dataset








