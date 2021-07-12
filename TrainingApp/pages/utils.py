import streamlit as st
from keras import models, layers, callbacks
from keras.models import save_model
import h5py
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tempfile import TemporaryFile
import base64


def build_model(input_num_genes, cell_type_num, customed_loss="categorical_crossentropy"):
    """建立线性神经网络模型
    ----------------------------------------------
    输入参数：
    input_num_genes：表达矩阵的基因种类数
    cell_type_num：预测的细胞种类数
    输出：
    神经网络模型
    ----------------------------------------------
    """
    nn = models.Sequential()
    nn.add(layers.Dense(200, activation="relu", input_shape=(input_num_genes, )))
    nn.add(layers.Dropout(0.4))
    nn.add(layers.Dense(100, activation="relu"))
    nn.add(layers.Dropout(0.2))
    nn.add(layers.Dense(cell_type_num, activation="softmax"))
    nn.compile(optimizer="RMSprop", loss=customed_loss, metrics=["accuracy"])
    return nn


@st.cache
def calc_class_weight(y):
    my_class_weight = class_weight.compute_class_weight("balanced", classes=np.unique(y), y=y)
    return dict(enumerate(my_class_weight))


def plot_acc(history):
    plt.scatter([i for i in range(1, len(history["accuracy"]) + 1)], history["accuracy"], label = "training_accuracy")
    plt.plot([i for i in range(1, len(history["val_accuracy"]) + 1)], history["val_accuracy"], label = "validation_accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    return plt


def plot_loss(history):
    plt.scatter([i for i in range(1, len(history["loss"]) + 1)], history["loss"], label = "training_loss")
    plt.plot([i for i in range(1, len(history["val_loss"]) + 1)], history["val_loss"], label = "validation_loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    return plt


def evaluate_one_type(model, type_name, total_dge, total_y, label_dict):
    cell_index = [i for i,_ in enumerate(total_y) if _ == type_name]
    type_index = label_dict[type_name]
    y_test = np.zeros((len(cell_index), len(label_dict)))
    y_test[:, type_index] = 1
    X_test = total_dge.iloc[cell_index ,:].values
    score = model.evaluate(X_test, y_test)
    return len(cell_index), score


@st.cache
def plot_test_score(test_cell_type, cell_nums, test_scores):
    data = pd.DataFrame({"cell_type" : test_cell_type, "cell_num" : cell_nums, "test_score" : test_scores})
    data = data.sort_values(by = ["test_score", "cell_num"])
    plt.barh([i for i in range(1, len(test_cell_type) + 1)], data["test_score"])
    str_cell_nums = [str(i) for i in data["cell_num"]]
    plt.yticks([i for i in range(1, len(test_scores) + 1)], [i+ "(" +j + ")" for i,j in zip(data["cell_type"], str_cell_nums)])
    plt.xlabel("Accuracy")
    plt.ylabel("Cell type")
    plt.title("Accuracy of each cell type")
    return data


def predict_one_type(model, type_name, total_dge, total_y, label_dict):
    cell_index = [i for i,_ in enumerate(total_y) if _ == type_name]
    type_index = label_dict[type_name]
    y_test = np.zeros((len(cell_index), len(label_dict)))
    y_test[:, type_index] = 1
    X_test = total_dge.iloc[cell_index ,:].values
    score = model.predict_classes(X_test)
    return len(cell_index), score


def save_label_dict(label_dict, file_name):
    label_df = pd.DataFrame({"id" : list(label_dict.values()), "celltype" : list(label_dict.keys())})
    label_df.index = label_df["id"]
    label_df.drop("id", inplace = True, axis = 1)
    label_df.to_csv(file_name)


@st.cache
def class_encode(cat):
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(cat)
    uniq_tag = np.unique(cat)
    mapping = pd.DataFrame({'id': encoder.transform(uniq_tag), 'celltype': uniq_tag})
    dummy_y = to_categorical(encoded_y)
    return dummy_y, mapping


class CustomCallback(callbacks.Callback):
    def __init__(self, emp1=None, cont=None, train_batch_size=None):
        super().__init__()
        self.emp1 = emp1
        self.cont = cont
        self.train_batch_size = train_batch_size
        self.plot_training = pd.DataFrame()
        self.temp = []
        # self.probar = []

    def on_epoch_begin(self, epoch, logs=None):
        with self.cont:
            c1, c2 = st.beta_columns([1, 7])
            with c1:
                st.text('Epoch ' + str(epoch) + ':')
            with c2:
                self.probar = st.empty()
                self.probar.progress(0.0)

    def on_epoch_end(self, epoch, logs=None):
        self.plot_training = self.plot_training.append(logs, ignore_index=True)
        self.emp1.line_chart(self.plot_training)
        self.probar.text('Done.')

    def on_batch_end(self, batch, logs=None):
        self.probar.progress(batch / self.train_batch_size)


def model2buffer(model):
    buffer = TemporaryFile()
    with h5py.File(buffer, 'w') as h:
        save_model(model, h, save_format='h5')
    return buffer


@st.cache(allow_output_mutation=True)
def downloader_buffer(fname, buffer):
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    label = fname.replace("_", r"\_")
    return f'<a href="data:file/h5;base64,{b64}" download="{fname}">{label}</a>'


def downloader_gene_list(fname, genes):
    b64 = base64.b64encode('\n'.join(genes).encode()).decode()
    label = fname.replace("_", r"\_")
    return f'<a href="data:file/txt;base64,{b64}" download="{fname}">{label}</a>'


def downloader(text, fname):
    b64 = base64.b64encode(text.encode()).decode()
    label = fname.replace("_", r"\_")
    return f'<a href="data:file/csv;base64,{b64}" download="{fname}">{label}</a>'