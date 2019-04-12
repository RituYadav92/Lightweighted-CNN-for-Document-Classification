### Train the model by running 
./train.py 

#### Optional arguments:
 ./train.py --help


### Evaluate Model

and, run below command 

./eval.py --eval_train --checkpoint_dir="./runs/trained_model/checkpoints/"

To use your own data, change the eval.py script to load your data.

### Test Model 
To get output for a set of tweets, make sure evaluate model is working , Then  run below :

python test.py --out_test --checkpoint_dir="./runs/trained_model/checkpoints/"

