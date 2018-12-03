# Usage
If you wanna to train a model, run the code below:   
python main.py --train_data data/train_vec.txt --dev_data data/dev_vec.txt --test_data data/test_vec.txt --max_epoch 30 --lr 0.0001 --dropout 0.2 --batch_size 128 --mode train  

And if you wanna predict your own data, change the --mode to 'test'. The sample data is in the directory ./data.

# Running environment
Python 3.6  
tensorflow 1.8  

# Ackonwledgement
This repository is a simple version of https://github.com/Determined22/zh-NER-TF. If you want to know more about BiLSTM-CRF on Chinese NER task, you should visit that repository.
