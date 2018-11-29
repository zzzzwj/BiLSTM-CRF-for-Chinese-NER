import os
import argparse
import numpy as np
import tensorflow as tf
from model import BiLSTM_CRF
from data import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--dev_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=40)
    parser.add_argument('--hidden_dim', type=int, default=300)
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
            generate_dictionary([args.train_data, args.test_data])
        else:
            generate_dictionary([args.train_data, args.dev_data, args.test_data])
    tag2label = pickle.load(open("data/tag2id.pkl", "rb"))
    word2id = pickle.load(open("data/word2id.pkl", "rb"))

    embeddings = np.random.uniform(-0.25, 0.25, (len(word2id), args.embedding_dim)).astype(np.float32)

    train_data = None
    dev_data = None
    test_data = None

    if args.mode == 'train':
        if args.train_data is None:
            print("Error! --train_data in train mode can't be ignored!")
            exit()
        else:
            train_data = read_file(args.train_data)
        if args.dev_data is not None:
            dev_data = read_file(args.dev_data)

    if args.mode == 'test':
        if args.test_data is None:
            print("Error! --test_data in test mode can't be ignored!")
            exit()
        else:
            test_data = read_file(args.test_data)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    print("Building network...")
    model = BiLSTM_CRF(batch_size = args.batch_size,
                       epoch = args.max_epoch,
                       hidden_dim = args.hidden_dim,
                       dropout = args.dropout,
                       lr = args.lr,
                       train_data = train_data,
                       dev_data = dev_data,
                       test_data = test_data,
                       embeddings = embeddings,
                       word2id = word2id,
                       tag2label = tag2label,
                       outputdir = args.model_path,
                       config = config)
    print("Done.")

    if args.mode == 'train':
        print('Start training...')
        model.train()
        print('Done.')

    elif args.mode == 'test':
        test_size = len(test_data)
        ckpt_file = tf.train.latest_checkpoint(os.path.join(args.model_path, 'checkpoints'))
        print("Predicting {} instances...".format(test_size))
        tags_res = model.predict(ckpt_file)
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