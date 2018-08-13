# encoding=utf-8
import sys

import tensorflow as tf

__all__ = ["metric_collect"]


def metric_collect(real_labels, predict_labels, metrics):
    try:
        assert len(real_labels) == len(predict_labels)
    except AssertionError:
        tf.logging.error("Error: predict tag seq num doesn't equal real tag seq!")
        sys.exit(0)
    for real_label, predict_label in zip(real_labels, predict_labels):
        metrics = _label_exist_check(real_label, metrics)
        metrics = _label_exist_check(predict_label, metrics)
        if real_label == predict_label:
            metrics[real_label]["real"] += 1
            metrics[real_label]["correct"] += 1
            metrics[real_label]["predict"] += 1
        elif real_label != predict_label:
            metrics[real_label]["real"] += 1
            metrics[predict_label]["predict"] += 1
    return metrics


def _label_exist_check(label, info):
    if not label:
        return info

    if label not in info:
        info[label] = {"real": 0, "predict": 0, "correct": 0}
    return info