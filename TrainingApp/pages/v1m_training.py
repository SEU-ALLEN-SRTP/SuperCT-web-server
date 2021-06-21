#%%

import keras
from keras import models, layers
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import pickle
import plotly
import plotly.plotly as py

## Load training data

X_train, Y_train, genes = pickle.load(open('/Users/WLin/Documents/AI_projects/gaomingxuan/ref/v1_training_data.pkl', 'rb'))
X_train = X_train.toarray()
Y_train = Y_train.toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=76)

## Build model

def build_model(input_num_genes, cell_type_num, customed_loss = "categorical_crossentropy"):
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
    nn.add(layers.Dense(200, activation = "relu", input_shape = (input_num_genes, )))
    nn.add(layers.Dropout(0.4))
    nn.add(layers.Dense(100, activation = "relu"))
    nn.add(layers.Dropout(0.2))
    nn.add(layers.Dense(cell_type_num, activation = "softmax"))
    nn.compile(optimizer = "rmsprop", loss = customed_loss, metrics = ["accuracy"])
    return nn

def calc_class_weight(y):
    my_class_weight = class_weight.compute_class_weight("balanced", np.unique(y), y)
    return my_class_weight

def plot_acc(history):
    plt.scatter([i for i in range(1, len(history["acc"]) + 1)], history["acc"], label = "training_accuracy")
    plt.plot([i for i in range(1, len(history["val_acc"]) + 1)], history["val_acc"], label = "validation_accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
def plot_loss(history):
    plt.scatter([i for i in range(1, len(history["loss"]) + 1)], history["loss"], label = "training_loss")
    plt.plot([i for i in range(1, len(history["val_loss"]) + 1)], history["val_loss"], label = "validation_loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

def evaluate_one_type(model, type_name, total_dge, total_y, label_dict):
    cell_index = [i for i,_ in enumerate(total_y) if _ == type_name]
    type_index = label_dict[type_name]
    y_test = np.zeros((len(cell_index), len(label_dict)))
    y_test[:, type_index] = 1
    X_test = total_dge.iloc[cell_index ,:].values
    score = model.evaluate(X_test, y_test)
    return len(cell_index), score

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

celltype_weights = calc_class_weight(np.argmax(Y_train, axis=1))
# celltype_weights = calc_class_weight(np.argmax(Y_train.toarray(), axis=1))
celltype_names = pd.read_csv('Data/MicroWell/id2type_news.csv', index_col=0)['celltype'].tolist()

n_cells, n_genes = X_train.shape
n_types = len(celltype_names)

nn = build_model(n_genes, n_types)

pickle.dump([genes, n_cells, n_genes, n_types, celltype_names], open('keras_parameters.pickle', 'wb'))

## Training

nn = build_model(n_genes, n_types)
history = nn.fit(X_train, Y_train, epochs=10, batch_size=1024,
                validation_split = 0.2, shuffle = True,
                class_weight = celltype_weights)


plot_acc(history.history)

plot_loss(history.history)

# Optimal training iteration
optimal_epochs = 5

nn = build_model(n_genes, n_types)
history = nn.fit(X_train, Y_train, epochs=optimal_epochs, batch_size = 1024,
                shuffle = True, class_weight = celltype_weights)

nn.save("./model/v1_5r.h5")

nn.evaluate(X_train, Y_train, batch_size=128, )

nn.evaluate(X_test, Y_test, batch_size=128, )
