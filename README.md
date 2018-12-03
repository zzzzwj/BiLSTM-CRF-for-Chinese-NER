# Usage
在目录下运行命令：  
python main.py --train_data data/train_vec.txt --dev_data data/dev_vec.txt --test_data data/test_vec.txt --max_epoch 30 --lr 0.0001 --dropout 0.2 --batch_size 128 --mode train  

模型存放于model_saved/checkpoint文件夹内  
当batchsize=128时，大约需要5GB显存  
当需要进行预测时仅需修改--mode为test即可  

# Running environment
Python 3.6  
tensorflow 1.8v  