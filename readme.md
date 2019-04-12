Code Forked from : https://github.com/cahya-wirawan/cnn-text-classification-tf

### Train the model by running 
./train.py 

#### Optional arguments:
 ./train.py --help


### Evaluate Model
To evaluate a pretrained model by me, download it from the link http://www.mediafire.com/folder/5n7i90i2kfaz8/trained_model
and keep in the 'runs' folder

and, run below command 

./eval.py --eval_train --checkpoint_dir="./runs/trained_model/checkpoints/"

To use your own data, change the eval.py script to load your data.

### Test Model 
To get output for a set of tweets, make sure evaluate model is working , Then  run below :

python test.py --out_test --checkpoint_dir="./runs/trained_model/checkpoints/"

