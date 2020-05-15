# ggc

This is a Python implementation for **Geometric Graph Construction (GGC)** from data features, as described in our paper:
 
Yifan Qian, Paul Expert, Pietro Panzarasa, and Mauricio Barahona, [Geometric graphs from data to aid classification tasks with graph convolutional networks](https://arxiv.org/abs/2005.04081).


Installation
------------

```python setup.py install```

Run the demo
------------
```
cd ggc
jupyter notebook demo.ipynb
```
Data
------------
Data sets can be found [here](https://github.com/haczqyf/ggc/tree/master/ggc/data). Each data set is a single csv-like file. Each row represents a sample. The first column is sample id, the last column is sample label and all the columns in the middle are features. Detailed descriptions of origins of data sets are described in the SI Appendix in our [paper](https://arxiv.org/abs/2005.04081). For the splits of training/validation/test sets, the first N1 rows consist of the traning set, the next N2 rows consist of the validation set, and the rest of N3 rows corresponds to the test set. The exact numbers of N1(~5% of samples), N2(~10% of samples) and N3(~85% of samples) are described in the SI Appendix in our [paper](https://arxiv.org/abs/2005.04081). The samples in the training set are evenly distributed across classes. Data can be loaded by using Kipf's code [here](https://github.com/tkipf/keras-gcn/blob/eb89564a0e865640c11283991685d80c84bc602a/kegra/utils.py#L15).

Cite
------------
Please cite our paper if you use this code in your own work:
```
@article{qian2020geometric,
  title={Geometric graphs from data to aid classification tasks with graph convolutional networks},
  author={Qian, Yifan and Expert, Paul and Panzarasa, Pietro and Barahona, Mauricio},
  journal={arXiv:2005.04081},
  year={2020}
}
```
