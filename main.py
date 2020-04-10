import argparse
import os
import tensorflow as tf

from data_loader import Data
from model import Model
import  shutil
parser = argparse.ArgumentParser(description='Unsupervised konwledge learning for HSI Classification')

parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--log',dest='log',default='log')
parser.add_argument('--model',dest='model',default='model')
parser.add_argument('--tfrecords',dest='tfrecords',default='tfrecords')
parser.add_argument('--data_name',dest='data_name',default='PaviaU')
parser.add_argument('--data_path',dest='data_path',default="../../dataset")

parser.add_argument('--use_lr_decay',dest='use_lr_decay',default=True)
parser.add_argument('--decay_rete',dest='decay_rete',default=0.90)
parser.add_argument('--decay_steps',dest='decay_steps',default=10000)
parser.add_argument('--learning_rate',dest='lr',default=0.001)
parser.add_argument('--train_num',dest='train_num',default=10) # intger for number and decimal for percentage
parser.add_argument('--cluster_batch',dest='cluster_batch',default=100)
parser.add_argument('--classification_batch',dest='classification_batch',default=90)
parser.add_argument('--cluster_num',dest='cluster_num',default=100)
parser.add_argument('--concate_way',dest='concate_way',default=1)
parser.add_argument('--fix_seed',dest='fix_seed',default=True)
parser.add_argument('--seed',dest='seed',default=666)
parser.add_argument('--test_batch',dest='test_batch',default=5000)
parser.add_argument('--epoch',dest='epoch',default=200001)
parser.add_argument('--cube_size',dest='cube_size',default=3)
parser.add_argument('--layers_num',dest='layers_num',default=9)

parser.add_argument('--save_decode_map',dest='save_decode_map',default=True)
parser.add_argument('--save_decode_segmap',dest='save_decode_segmap',default=True)
parser.add_argument('--del_tfrecords',dest='del_tfrecords',default=True)

parser.add_argument('--load_model',dest='load_model',default=False)


args = parser.parse_args()
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)


def main():

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'# set your GPU ID
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False

    for i in range(1):
        args.id = str(i)
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            args.result = os.path.join(args.result,args.id)
            args.log = os.path.join(args.log, args.id)
            args.model = os.path.join(args.model, args.id)
            args.tfrecords = os.path.join(args.tfrecords, args.id)
            if not os.path.exists(args.model):
                os.mkdir(args.model)
            if not os.path.exists(args.log):
                os.mkdir(args.log)
            if not os.path.exists(args.result):
                os.mkdir(args.result)
            if not os.path.exists(args.tfrecords):
                os.mkdir(args.tfrecords)

            dataset = Data(args)
            dataset.read_data_cluster()
            dataset.read_data()
            model = Model(args,sess)
            if not args.load_model:
                model.train(dataset)
            else:
                model.load(args.model)
                model.test2(dataset)

            if args.save_decode_map:
                model.save_decode_map(dataset)
            if args.save_decode_segmap:
                model.save_decode_seg_map(dataset)
            if args.del_tfrecords:
                shutil.rmtree(args.tfrecords)
            args.result = 'result'
            args.log = 'log'
            args.tfrecords = 'tfrecords'
            args.model = 'model'

if __name__ == '__main__':
    main()
