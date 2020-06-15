## Light-Weighted CNN for Text Classification

### Dataset Used
Tobacco-3482

Categories in the dataset are:

```['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']```

### Train Model
#### To train on Optimized Text CNN
```python ADAM_optmized_train.py```

#### Lightweight Text CNN
```python singleADAM_LW_train.py```

#### Lightweight TextCNN with Dual Optimizer
Switches from Adam to SGD when a triggering condition is satisfied.
```python SWAT_LW_train.py```

#### Optional arguments:
 ```python train.py --help```


### Evaluate Model

To evaluate, run below command 

```python eval.py --eval_train --checkpoint_dir="./runs/trained_model/checkpoints/"```

To use your own data, change the eval.py script to load your data.

### Test Model 
To prediction on new test data, make sure evaluate model is working , Then run below :

```python test.py --out_test --checkpoint_dir="./runs/trained_model/checkpoints/" --test_dir="path to test data"```

### Link to the paper
For more details please go through my paper at link: https://arxiv.org/pdf/2004.07922.pdf

### References :
1. https://github.com/dennybritz/cnn-text-classification-tf
