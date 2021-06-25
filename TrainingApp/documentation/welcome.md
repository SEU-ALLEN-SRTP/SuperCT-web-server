# Training Guide :surfer:

---
This app is mainly a set of Python scripts, having the following directory structure:

```
├── pages
│   ├── predict.py
│   ├── v1m_training.py
│   ├── v1h_training.py
│   ├── v2m_training.py
│   └── welcome.py
├── documentation
│   ├── sidebar_header.md
│   └── welcome.md
└── main.py
```

Use `streamlit run main.py` in the command line interface to start up a server on `localhost:8501` by default. 
Just make sure all the dependencies are installed properly before you go.

Since we have 3 models and each of them assumes a slightly different training strategy, we therefore
provide you with three training modes in this app.

## Training Modes

---

### v1m
The v1m model is the most primary model given by the [paper](https://dx.doi.org/10.1093%2Fnar%2Fgkz116). It's trained
from scratch using the MCA dataset, outputing 30 cell types.

### v1h
Based on the v1m model, v1h model further takes in human PBMC dataset as for transfer learning.

### v2m
To change the output style of the v1m model, v2m training freezes the first hidden layer's parameters in v1m
and change the output number from 30 to 38 before retraining.

---

_Zuohan Zhao,  
Southeast University,  
Nanjing  
2021/6/21_