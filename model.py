import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import unit
class Model():

    def __init__(self,args,sess):
        self.sess = sess
        self.result = args.result
        self.data_name = args.data_name
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.class_num = int(info['class_num'])
        self.data_gt = info['data_gt']
        self.log = args.log
        self.model = args.model
        self.cube_size = args.cube_size
        self.data_path = args.data_path
        self.epoch = args.epoch
        self.tfrecords = args.tfrecords
        self.global_step = tf.Variable(0,trainable=False)
        self.training = tf.placeholder(bool)
        self.cluster_num = args.cluster_num
        self.train_num = args.train_num
        self.classification_batch = args.classification_batch
        self.cluster_batch = args.cluster_batch
        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr
        
        self.concate_way = args.concate_way
        self.af = tf.nn.relu
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube_size,self.cube_size,self.dim))
        self.label = tf.placeholder(dtype=tf.int64, shape=(None,1))
        self.p_label = tf.placeholder(dtype=tf.int64,shape=(None,1))
        if args.layers_num == 3:
            self.classifer_share = unit.classifer_share3d_3
        elif args.layers_num == 6:
            self.classifer_share = unit.classifer_share3d_6
        elif args.layers_num == 9:
            self.classifer_share = unit.classifer_share3d_9
        elif args.layers_num == 12:
            self.classifer_share = unit.classifer_share3d_12
        self.feature = self.classifer_share(self.image,self.training,reuse=False,cube_size=self.cube_size)
        self.feature_cluster = self.feature[:-self.classification_batch]
        self.feature_classification = self.feature[-self.classification_batch:]

        self.cluster_fc,self.cluster_label = unit.classifer_cluster(self.feature_cluster,self.training,self.cluster_num, reuse=False)
        self.cc1_fc,self.cc1_label = unit.classifer_cluster(self.feature_classification,self.training,self.cluster_num,reuse=True)
        self.classification_fc,self.classification_label = unit.classifer_classification(self.feature_classification,self.training,self.class_num,reuse=False)
        self.cc2_fc,self.cc2_label = unit.classifer_classification(self.feature_cluster,self.training,self.class_num,reuse=True)
        self.combine_label_classification = unit.classifer_combine(self.classification_fc,self.cc1_fc,self.concate_way,self.class_num,self.training,reuse=False)
        self.combine_label_cluster = unit.classifer_combine(self.cc2_fc,self.cluster_fc,self.concate_way,self.class_num,self.training,reuse=True)

        self.cluster_fc_test,_ = unit.classifer_cluster(self.feature,self.training,self.cluster_num, reuse=True)
        self.classification_fc_test,self.pre_label_test = unit.classifer_classification(self.feature,self.training,self.class_num,reuse=True)
        self.pre_label_test_combine = unit.classifer_combine(self.classification_fc_test,self.cluster_fc_test,self.concate_way,self.class_num,self.training,reuse=True)

        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.log),graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=5)

    def loss(self):
        with tf.variable_scope('loss'):
            loss_cross_entropy_a = tf.losses.sparse_softmax_cross_entropy(labels=self.p_label,
                                                                          logits=self.cluster_label,
                                                                          scope='cluster_loss_cross_entropy')
            self.loss_cross_entropy_cluster = tf.reduce_mean(loss_cross_entropy_a)

            loss_cross_entropy_b = tf.losses.sparse_softmax_cross_entropy(labels=self.label,
                                                                          logits=self.classification_label,
                                                                          scope='classification_loss_cross_entropy')
            self.loss_cross_entropy_classification = tf.reduce_mean(loss_cross_entropy_b)

            loss_cross_entropy_c = tf.losses.sparse_softmax_cross_entropy(labels=self.label,
                                                                          logits=self.combine_label_classification,
                                                                          scope='combine1_loss_cross_entropy')
            self.loss_cross_entropy_fusion_classification = tf.reduce_mean(loss_cross_entropy_c)
            loss_cross_entropy_d = tf.losses.sparse_softmax_cross_entropy(labels=self.p_label,
                                                                          logits=self.combine_label_cluster,
                                                                          scope='combine1_loss_cross_entropy')
            self.loss_cross_entropy_fusion_cluster = tf.reduce_mean(loss_cross_entropy_d)

            self.loss_total = self.loss_cross_entropy_cluster + self.loss_cross_entropy_classification + self.loss_cross_entropy_fusion_classification\
                #+self.loss_cross_entropy_fusion_cluster
            tf.summary.scalar('loss_cross_entropy_cluster',self.loss_cross_entropy_cluster)
            tf.summary.scalar('loss_cross_entropy_classification',self.loss_cross_entropy_classification)
            tf.summary.scalar('loss_cross_entropy_combine1', self.loss_cross_entropy_fusion_classification)
            tf.summary.scalar('loss_cross_entropy_combine2', self.loss_cross_entropy_fusion_cluster)
            tf.summary.scalar('loss_total',self.loss_total)
        tf.summary.scalar('lr', self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


    
    def load(self, checkpoint_dir):
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train(self,dataset):
        train_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'train_data.tfrecords'), type='train')
        cluster_dataset = dataset.data_parse_cluster(os.path.join(self.tfrecords, 'cluster_data.tfrecords'), type='train')

        init = tf.global_variables_initializer()
        self.sess.run(init)
        oa_list,aa_list,kappa_list,matrix_list,ac_llist = list(),list(),list(),list(),list()
        best_acc = 0
        for i in range(1,self.epoch):
            train_data,train_label,train_p_label = self.sess.run(train_dataset)
            cluster_data,cluster_label,cluster_p_label = self.sess.run(cluster_dataset)
            fusion_data = np.concatenate((cluster_data,train_data),axis=0)
            l,lu,lc,lf,_,summery,lr= self.sess.run([self.loss_total,self.loss_cross_entropy_cluster,self.loss_cross_entropy_classification,
                                                 self.loss_cross_entropy_fusion_classification,
                                        self.optimizer,self.merged,self.lr],
                                        feed_dict={self.image:fusion_data,
                                                  self.label:train_label,self.p_label:cluster_label,self.training:True})
            if i % 1000 == 0:
                print('step %d, loss_cluster %f ,loss_classification %f,loss_fusion %f,loss_total %f lr %f'%(i,lu,lc,lf,l,lr))
            if i % 5000 == 0:
                # _ = self.test1(dataset)
                oa,aa,kappa,ac_list,matrix = self.test2(dataset)
                if oa > best_acc:
                    best_acc = oa
                    self.saver.save(self.sess,os.path.join(self.model,self.model_name),global_step=self.global_step)
                    print('best model saved...')
                print('best accuracy:%f' % (best_acc))

                oa_list.append(oa)
                aa_list.append(aa)
                kappa_list.append(kappa)
                ac_llist.append(ac_list)
                matrix_list.append(matrix)
                sio.savemat(os.path.join(self.result, 'result_list.mat'),
                            {'oa': oa_list, 'aa': aa_list, 'kappa': kappa_list,'ac_list':ac_llist, 'matrix': matrix_list,'best_acc':best_acc})
            self.summary_write.add_summary(summery,i)

    def test1(self,dataset):
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.float64)
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                pre_label = self.sess.run(tf.argmax(tf.nn.softmax(self.pre_label_test),1), feed_dict={self.image:test_data,\
                    self.label:test_label,self.training:False})
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)

        sio.savemat(os.path.join(self.result, 'result1.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
        return oa,aa,kappa,ac_list,matrix

    def test2(self,dataset):
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.float64)
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                # print(test_data.shape,test_label.shape)
                pre_label = self.sess.run(tf.argmax(tf.nn.softmax(self.pre_label_test_combine),1), feed_dict={\
                    self.image:test_data,self.label:test_label,self.training:False})
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)

        sio.savemat(os.path.join(self.result, 'result2.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
        return oa,aa,kappa,ac_list,matrix

    def save_decode_map(self,dataset):
        map_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'map_data.tfrecords'), type='map')
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        # data_gt = info['data_gt']
        data_gt = info['data_gt'][::-1]
        fig, _ = plt.subplots()
        height, width = data_gt.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0,0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(data_gt, cmap='jet')
        plt.savefig(os.path.join(self.result, 'groundtrouth_' + self.data_name + '.png'), format='png', dpi=800)
        plt.close()
        print('Groundtruth map get finished')
        de_map = np.zeros(data_gt.shape,dtype=np.int32)
        # print(data_gt.shape)
        try:
            while True:
                map_data,pos = self.sess.run(map_dataset)
                pre_label = self.sess.run(self.pre_label_test_combine, feed_dict={self.image:map_data,self.training:False})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    [r,c]=pos[i]
                    de_map[r,c] = pre_label[i] + 1
        except tf.errors.OutOfRangeError:
            print("map draw end!")
        de_map = de_map[::-1]
        fig, _ = plt.subplots()
        height, width = de_map.shape
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map_' + self.data_name + '.png'), format='png', dpi=800)  # bbox_inches='tight',pad_inches=0)
        plt.close()
        print('decode map get finished')

    def save_decode_seg_map(self,dataset):
        map_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'map_seg_data.tfrecords'), type='map_seg')
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        data_gt = info['data_gt']
        de_map = np.zeros(data_gt.shape,dtype=np.int32)
        try:
            while True:
                map_data,pos = self.sess.run(map_dataset)
                pre_label = self.sess.run(self.pre_label_test_combine, feed_dict={self.image:map_data,self.training:False})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    [r,c]=pos[i]
                    de_map[r,c] = pre_label[i] + 1
        except tf.errors.OutOfRangeError:
            print("test end!")
        de_map = de_map[::-1]
        fig, _ = plt.subplots()
        height, width = de_map.shape
        fig.set_size_inches(width/100.0, height/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map_seg'+self.data_name+'.png'),format='png',dpi=600)#bbox_inches='tight',pad_inches=0)
        plt.close()
        print('seg decode map get finished')