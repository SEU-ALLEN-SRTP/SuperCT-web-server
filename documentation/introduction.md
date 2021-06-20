---

## Introduction
This web server implements the supervised classification of scRNA data algorithm 
[SuperCT](https://github.com/weilin-genomics/SuperCT), powered by [Streamlit](https://streamlit.io).
Your can learn more about the algorithm in this [paper](https://dx.doi.org/10.1093%2Fnar%2Fgkz116).

SuperCT uses artificial neural networks to predict cell types based on single-cell DGE data, 
presuming the input data are all binary, meaning genes are either expressed or not. 
The features used to predict cell types are a special set of genes that are applicable to both human and mouse data. 
The paper describing the algorithm is published in [here](https://academic.oup.com/nar/article/47/8/e48/5364134).

This server enables you to use the model to get the prediction of your data 
online without installing all the requirements, and returns fine plots for your data visualization.

---

## Data format

For questions of how the input and output of the model should look like, 
refer to the `example` data in this app or the github repo of SuperCT for more details. 
You can also refer to the examples illustrated below.

The examples are all binary, but you __can__ still upload numeric data because there's
__an inner binarization step__ before prediction and training, and __should__ if you want get
a proper UMAP or tSNE map.

### Input table:

This is the kind of table that the model takes.
Note that you can __upload a numeric__ one instead.

|     | Cell 1  | Cell 2  | Cell 3 | ... |
|:---:|:-------:|:-------:|:------:|-----|
| Gene1  |   0   |   0   |   1   | ... |
| Gene2  |   0   |   1   |   0   | ... |
| Gene3  |   0   |   0   |   1   | ... |
|   ...  |   ... |   ... |   ... | ... |

### Output table:

The prediction of the network is actually numeric, which means there should be a mapping table
for you to turn the numbers into meaningful classes.

As for our default model, we've already 
prepared a default mapping that gives you a table as such.

| cell_id | pred_type |
|:-------:|:---------:|
| Cell 1  | Adipocyte |
| Cell 2  | Stomach   |
| Cell 3  | Radial    |
| ...     | ...       |

### Some Notes
* Use Gene symbols as features.

---

## Responsible

Peng Xie (Author), Zuohan Zhao (Developer)

---

__Have fun!__

_Zuohan Zhao,  
Southeast University,  
Nanjing  
2021/6/16_
