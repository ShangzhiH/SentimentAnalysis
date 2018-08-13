# encoding=utf-8

"""
该文件提供一些与数据处理无关的工具类函数
"""
import os
import pickle

import tensorflow as tf


def print_flags(flags_to_print, wrapped=False):
    if wrapped:
        flags_dict = flags_to_print.__dict__["__wrapped"].__dict__["__flags"]
        for k, v in flags_dict.items():
            tf.logging.info("{}:\t{}".format(k.ljust(15), v.value))
    else:
        flags_dict = flags_to_print.__dict__['__flags']
        for k, v in flags_dict.items():
            tf.logging.info("{}:\t{}".format(k.ljust(15), v))


def save_flags(flags_to_save, path, wrapped=False):
    with tf.gfile.GFile(path, "wb") as f_w:
        if wrapped:
            flags_dict = flags_to_save.__dict__['__wrapped'].__dict__['__flags']
        else:
            flags_dict = flags_to_save.__dict__['__flags']
        pickle.dump(flags_dict, f_w)


def load_flags(path):
    with tf.gfile.GFile(path, "rb") as f:
        flags_dict = pickle.load(f)
        flags = tf.app.flags.FLAGS
        for k, v in flags_dict.items():
            flags.__dict__['__flags'][k] = v
    return flags


def unicode_open(filepath):
    with tf.gfile.GFile(filepath, "r") as f:
        for line in f:
            if type(line) == str:
                line = line.decode("utf-8")
            yield line


def create_model(session, Model_class, path, load_vec, config, id_to_char, logger):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value())
            emb_weights = load_vec(logger, config["emb_file"],id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def export_model(sess, model, path, version, logger, id_to_tag, char_to_id):
    export_path = os.path.join(path, str(version))
    asset_path = os.path.join(path)
    if tf.gfile.IsDirectory(export_path):
        tf.gfile.DeleteRecursively(export_path)

    with tf.gfile.GFile(os.path.join(asset_path, "char_to_id.csv"), "w") as file_w:
        for key, value in char_to_id.iteritems():
            file_w.write(u"%s\t%s\n" % (key, value))
    with tf.gfile.GFile(os.path.join(asset_path, "id_to_tag.csv"), "w") as file_w:
        for key, value in id_to_tag.iteritems():
            file_w.write(u"%s\t%s\n" % (key, value))

    #char_to_id_asset = tf.constant("model/char_to_id.csv")
    #id_to_tag_asset = tf.constant("model/id_to_tag.csv")
    #tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, char_to_id_asset)
    #tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, id_to_tag_asset)
    #char_to_id_tensor = tf.Variable("char_to_id.csv", name = "char_to_id_tensor", trainable=False, collections=[])
    #assign_char_to_id_op = char_to_id_tensor.assign("char_to_id.csv")

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.char_inputs)
    tensor_info_logits = tf.saved_model.utils.build_tensor_info(model.logits)
    tensor_info_trans = tf.saved_model.utils.build_tensor_info(model.trans)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(model.dropout)

    predict_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'sentences': tensor_info_x, 'dropout': tensor_info_dropout},
            outputs = {'logits': tensor_info_logits, 'trans': tensor_info_trans},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map = {
            'predict_tags':
                predict_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature,
        },
        legacy_init_op = legacy_init_op,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
    )
    builder.save()
    logger.info('Done exporting')


