# encoding=utf-8
import tensorflow as tf
import pickle

from model_dataset import DatasetMaker
from utils import convert_from_keras_to_tensorflow, unicode_open

if __name__ == "__main__":
    keras_model = "data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression"
    export_dir = "/Users/dianping/Documents/work/project/nlp/SentimentAnalysis/General/SentimentAnalysis/data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression_tf"
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    convert_from_keras_to_tensorflow(keras_model, export_dir)

    char_to_id = {}
    id_to_char = {}
    for line in unicode_open("data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression_word_pos_dict"):
        try:
            word, id = line[:-1].split(u"\t")
            char_to_id[word] = int(id)
            id_to_char[int(id)] = word
        except Exception:
            print(line)

    print(len(char_to_id))
    print(len(id_to_char))
    with tf.gfile.GFile("data/bidirectional_lstm_model_200_msl_scbm_max_3_epoch_3_mtf_300_dim_comment_all_data_w2v_embed_fine_tuned_comment_data_regression_word_pos_dict.pkl", "wb") as f_w:
        pickle.dump([char_to_id, id_to_char], f_w)