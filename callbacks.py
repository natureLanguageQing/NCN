# encoding: utf-8

# author: nlqing
# contact: ygq624576166@163.com

# file: callbacks.py
# time: 2019-05-22 15:00
import macros
from tasks.base_model import BaseModel
from seqeval import metrics as seq_metrics
from sklearn import metrics
from tensorflow.python import keras


class EvalCallBack(keras.callbacks.Callback):

    def __init__(self, kash_model: BaseModel, valid_x, valid_y, step=5, batch_size=256):
        """
        Evaluate callback, calculate precision, recall and f1
        Args:
            kash_model: the NCN model to evaluate
            valid_x: feature data
            valid_y: label data
            step: step, default 5
            batch_size: batch size, default 256
        """
        super(EvalCallBack, self).__init__()
        self.kash_model = kash_model
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.step = step
        self.batch_size = batch_size
        self.logs = {}

        self.average = 'weighted'

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step == 0:
            y_pred = self.kash_model.predict(self.valid_x, batch_size=self.batch_size)

            if self.kash_model.task == macros.TaskType.LABELING:
                y_true = [seq[:len(y_pred[index])] for index, seq in enumerate(self.valid_y)]
                precision = seq_metrics.precision_score(y_true, y_pred)
                recall = seq_metrics.recall_score(y_true, y_pred)
                f1 = seq_metrics.f1_score(y_true, y_pred)
            else:
                y_true = self.valid_y
                precision = metrics.precision_score(y_true, y_pred, average=self.average)
                recall = metrics.recall_score(y_true, y_pred, average=self.average)
                f1 = metrics.f1_score(y_true, y_pred, average=self.average)

            self.logs[epoch] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"\nepoch: {epoch} precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}")


if __name__ == "__main__":
    print("Hello world")
    config_path = '/Users/nlqing/Desktop/python/NCN/tests/test-data/bert/bert_config.json'
    check_point_path = '/Users/nlqing/Desktop/python/NCN/tests/test-data/bert/bert_model.ckpt'
