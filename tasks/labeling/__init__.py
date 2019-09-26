# encoding: utf-8

# author: NLQing
# contact: ygq624576166@163.com


# file: __init__.py
# time: 2019-05-20 11:34

from NCN.tasks.labeling.models import CNN_LSTM_Model

from NCN.tasks.labeling.models import BiLSTM_Model
from NCN.tasks.labeling.models import BiLSTM_CRF_Model

from NCN.tasks.labeling.models import BiGRU_Model
from NCN.tasks.labeling.models import BiGRU_CRF_Model

if __name__ == "__main__":
    print("Hello world")
