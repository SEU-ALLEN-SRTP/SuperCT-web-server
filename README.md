# SuperCT Web Server
This web server implements the supervised classification of scRNA data algorithm, [SuperCT](https://github.com/weilin-genomics/SuperCT), powered by Dash.

## Introduction

SuperCT uses artificial neural networks to predict cell types based on single-cell DGE data, presuming the input data are all binary, meaning genes are either expressed or not. The features used to predict cell types are a special set of genes that are applicable to both human and mouse data. The paper describing the algorithm is published in [here](https://academic.oup.com/nar/article/47/8/e48/5364134).

This server enables you to use the model to get the prediction of your data online without installing all the requirements, and returns fine plots for your data visualization.

## Data format

For questions of how the input and output should look like, see the `example` folder in this project or the github project of SuperCT for more details. Or see the examples below.

### Input table:

|   | Cell 1  | Cell 2  | Cell 3 | ... |
|---|:-------:|:-------:|:------:|-----|
| Gene1  |   0   |   0   |   1   | ... |
| Gene2  |   0   |   1   |   0   | ... |
| Gene3  |   0   |   0   |   1   | ... |
| Gene4  |   0   |   0   |   0   | ... |

### Output table:

| cell_id | pred_type |
|:-------:|:---------:|
| Cell 1  | Adipocyte |
| Cell 2  | Stomach   |
| Cell 3  | Radial    |
| ...     | ...       |

__Note__: Use Gene symbols as features.

Have fun!

by Zuohan Zhao, 

Southeast University,

2021/6/16