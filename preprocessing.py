# encoding=utf-8
import os

import tensorflow as tf

from utils import unicode_open
from data_utils import replace_html, full_to_half

tf.logging.set_verbosity(tf.logging.INFO)

def concat_directory_files(filepath, dst_file):
    with tf.gfile.GFile(dst_file, "w") as f_w:
        for filename in tf.gfile.ListDirectory(filepath):
            if "part-" not in filename:
                continue
            tf.logging.info(u"Processing file {}".format(filename))
            for line in unicode_open(os.path.join(filepath, filename)):
                if line.strip() == "":
                    continue
                reviewbody = line.strip()[:-2]
                star = line.strip()[-2] + line.strip()[-1]
                if star == "10":
                    score = "-1"
                else:
                    score = "1"
                reviewbody = _process_line(reviewbody)
                f_w.write(u"{}\t{}\n".format(reviewbody, score))

def _process_line(reviewbody):
    return full_to_half(replace_html(reviewbody))

if __name__ == "__main__":
    filepath = "viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/huangshangzhi/sentiment_analysis/data/3000w_train.csv/"
    dst = "viewfs://hadoop-meituan/ghnn01/user/hadoop-poistar/huangshangzhi/sentiment_analysis/data/train_2000wdp_1000wmt.csv"
    concat_directory_files(filepath, dst)