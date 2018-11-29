import tensorflow as tf
import os, argparse, time
from model import BiLSTM_CRF
from data import *


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str)
parser.add_argument('--dev_data', type=str, default=None)
parser.add_argument('--test_data', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--CRF', action = 'store_true')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--model_path', type=str, default='model_save')
args = parser.parse_args()

if not os.path.exists('data'):
    os.mkdir('data')
if not (os.path.exists('data/tag2id.pkl') and os.path.exists('data/word2id.pkl')):
    if args.dev_data is None:
        GenerateDictionary([args.train_data, args.test_data])
    else:
        GenerateDictionary([args.train_data, args.dev_data, args.test_data])
tag2label = pickle.load(open("data/tag2id.pkl", "rb"))
word2id = pickle.load(open("data/word2id.pkl", "rb"))

# random embedding words
embeddings = random_embedding(word2id, args.embedding_dim)

if args.mode == 'train':
    train_data = read_corpus(args.train_data)
    if args.dev_data is None:
        dev_data = None
    else:
        dev_data = read_corpus(args.dev_data)
elif args.mode == 'test':
    test_data = read_corpus(args.test_data)
    test_size = len(test_data)

if not os.path.exists(args.model_path):
    os.mkdir(args.model_path)

if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, args.model_path, config=config)
    model.build_graph()
    print('Start training...')
    model.train(train=train_data, dev=dev_data)
    print('Done.')

elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(os.path.join(args.model_path, 'checkpoints'))
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, args.model_path, config=config)
    model.build_graph()
    print("Predicting {} instances...".format(test_size))
    tags_res = model.predict(ckpt_file, test_data)
    print("Done.")

    print("Write result to file data/result.txt.")
    with open("data/result.txt", "wt") as f:
        for i in range(test_size):
            sentence = test_data[i][0]
            tags = tags_res[i]
            for _, tag in zip(sentence, tags):
                f.write(tag + " ")
            f.write("\n")
    print("Done.")