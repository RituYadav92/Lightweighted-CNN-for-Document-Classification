### Train the model by running 
```python train.py ```

### Evaluate Model

```python eval.py --eval_train --checkpoint_dir="./runs/trained_model/checkpoints/"```

To use your own data, change the eval.py script to load your data.

### Test Model 

```python test.py --out_test --checkpoint_dir="./runs/trained_model/checkpoints/"```

