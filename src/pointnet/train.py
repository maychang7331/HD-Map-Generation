import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import socket
import provider
import tf_util
from model import *
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='logdso8(kp0.6 valid_rnd)', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=str, default=str('9_16'), help='Which area to use for test, option: 1-48 [default: 9_16]')
FLAGS = parser.parse_args()


GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): 
    os.makedirs(LOG_DIR)

LOG_DIR = os.path.join(BASE_DIR,FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def      # 將字串轉化成命令再執行（類似cmd功能）
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 5         ####################################### 

# 用來計算Batch Normalization 的Decay 參數(即decay參數也隨著訓練逐漸decay)
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname() # 獲取當前主機的主機名

ALL_FILES = provider.getDataFiles(os.path.join(BASE_DIR,'data/shalun_h5/dso8/all_h5files.txt'))
block_filelist = [line.rstrip() for line in open(os.path.join(BASE_DIR,'data/shalun_h5/dso8/block_filelist.txt'))]

# load All data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)   # (Total blocks, 4096, 9): X Y X" Y" Z I' R' G' B'
data_batches = data_batches[:, :, 2:6]              # (Total blocks, 4096, 4): 只放 X" Y" Z I'
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

# left_block_idxs = []
# for i in range(data_batches.shape[0]):
#     count_label0 = (label_batches[i] == 0).sum()
#     if count_label0 != MAX_NUM_POINT:
#         left_block_idxs.append(i)
# print(len(left_block_idxs))

startnum = int(FLAGS.test_area.split('_')[0])
endnum = int(FLAGS.test_area.split('_')[-1])

train_idxs = []
test_idxs = []

split_percentage = 0.2
bool_arr = np.random.choice(a=[True, False], size=len(data_batches), p=[split_percentage, 1-split_percentage])
(unique, counts) = np.unique(bool_arr, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

"""<隨機取20%個block區分train和test(valid)>
"""
for idx, isTest in enumerate(bool_arr):
    if isTest :
        test_idxs.append(idx)
    else:
        train_idxs.append(idx)

"""<用las區分train和test(valid)>

for i, district_name in enumerate(block_filelist):
    isTest = False
    for j in range(startnum, endnum+1):
        test_area = 'pt_'+ str(j)
        if test_area in district_name:
            isTest = True
            break    
        else:
            continue
    if isTest :
        test_idxs.append(i)
    else:
        train_idxs.append(i)
"""

train_data = data_batches[train_idxs, ...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs, ...]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)

NUM_FEATURES = train_data.shape[-1]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,     # Base learning rate. (初始學習率)
                        batch * BATCH_SIZE,     # Current index into the dataset.
                        DECAY_STEP,             # Decay step. (每100步衰減一次 DECAY_STEPS = 100)
                        DECAY_RATE,             # Decay rate. (衰減率)
                        staircase=True)         # 使用階梯式的衰減方式 https://www.itread01.com/content/1545386432.html
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!! 若0.00001，則設0.0001 (防止刚开始时学习速率过慢)
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                        BN_INIT_DECAY,
                        batch*BATCH_SIZE,       # global_step可以通过设置一个常量tensor来指定
                        BN_DECAY_DECAY_STEP,                                                                                            
                        BN_DECAY_DECAY_RATE,
                        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1-bn_momentum)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    return bn_decay

def train():
    tic = datetime.now()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    with tf.Graph().as_default():       # as_default:作為預設圖使用
        with tf.device('/gpu:' + str(GPU_INDEX)):
            # 讓 computational graph 保留輸入欄位的節點，其允許實際的輸入值留到後來再指定                                       
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)       
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. global step(總訓練步數) 參數初始化為0, 每次自動加1
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)      # 这个batch用来设置glob_step # 透過 tf.Variable(0)設定參數初始值定為0（這樣可以讓參數可以進行訓練）
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay) # https://blog.csdn.net/hongxue8888/article/details/79753679

            # Get model and loss
            # 調用model.py中的get_model(), 由get_model()可知pred維度為B*N*5, 5為channel數, 對應13個分類標籤
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay) 
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            # tf.argmax(pred, 2)返回pred C 這個維度的最大值索引
            # tf.equal()比較兩個張量對應位置是否相等,返回相同维度的bool值矩阵
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session 配置session運行參數
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 讓tensorflow 在運行過程中動態申請顯存, 避免過多的顯存佔用
        config.allow_soft_placement = True      # 當指定的設備不存在时，允許選擇一个存在的設備運行。比如gpu不存在，自動降到cpu上運行
        config.log_device_placement = True      # 在终端打印出各項操作是在哪个設備上運行的
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'valid'))

        # Init variables 初始化參數開始訓練
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)    # 用來訓練一個epoch
            eval_one_epoch(sess, ops, test_writer, epoch)      # 用來運行每一个epoch後evaluate在測試集的accuracy和loss。每10个epoch保存1次模型。
            
            # Save the variables to disk.
            if (epoch+1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
    toc = datetime.now()
    log_string('Elapsed time: %f seconds' % (toc-tic).total_seconds())


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE   # 計算在指定BATCH_SIZE下，訓練1个epoch 需要幾個mini-batch訓練ㄋ
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    # 在一个epoch中逐個mini-batch訓練直至遍歷完一遍訓練集。計算總分類正確數total_correct和已遍歷樣本數total_senn，總損失loss_sum
    for batch_idx in range(num_batches):
        if batch_idx % 5 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])    
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('(epoch) mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('(epoch) accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    # confusion matrix
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    diagonal = np.zeros([1, NUM_CLASSES])
    gt_total = np.zeros([NUM_CLASSES, 1])
    pred_total = np.zeros([1, NUM_CLASSES])
    precision =  np.zeros([1, NUM_CLASSES])
    recall = np.zeros([1, NUM_CLASSES])
    f1_score = np.zeros([1, NUM_CLASSES])
    
    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1  ####################################################
                total_correct_class[l] += (pred_val[i-start_idx, j] == l) #############################
                confusion_matrix[l, pred_val[i-start_idx, j]] +=1
    
    log_string('valid mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('valid accuracy: %f'% (total_correct / float(total_seen)))
    log_string('valid avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    diagonal = np.diag(confusion_matrix)
    gt_total = np.sum(confusion_matrix, axis=1)
    pred_total = np.sum(confusion_matrix, axis=0)
    precision = diagonal / pred_total
    recall = diagonal / gt_total
    f1_score = 2*precision*recall/(precision+recall)

    # 增加最後一欄 shape(5,) -> shape(6,)
    precision = np.append(precision, np.mean(precision))
    recall = np.append(recall, np.mean(recall))
    pred_total = np.append(pred_total, np.sum(pred_total))
    f1_score = np.append(f1_score, np.mean(f1_score))

    # shape(6,) -> shape(6,1)
    precision = precision.reshape(6, 1)
    recall = recall.reshape(6, 1)
    pred_total = pred_total.reshape(6, 1)
    f1_score = f1_score.reshape(6, 1)

    # 合併整個 confusion matrix
    confusion_matrix = np.concatenate((confusion_matrix, gt_total[:, None]), axis=1)    # 加到最後一行
    confusion_matrix = np.concatenate((confusion_matrix, pred_total.T), axis=0)         # 加到最後一列
    confusion_matrix = np.concatenate((confusion_matrix, precision.T), axis=0)
    confusion_matrix = np.concatenate((confusion_matrix, recall.T), axis=0)
    confusion_matrix = np.concatenate((confusion_matrix, f1_score.T), axis=0)

    df = pd.DataFrame(confusion_matrix, 
                        columns=['other', 'traffic island', 'sign', 'signal', 'pole', 'Total'],
                        index=['other', 'traffic island', 'sign', 'signal', 'pole', 'Total', 'Precision', 'Recall', 'F1 Score'])
    df.T.round(decimals=dict(zip(df.index, [0, 0, 0, 0, 0, 0, 2, 2, 2]))).T
    df = df.astype(object)
    print(df)
    
    if (epoch+1) == MAX_EPOCH:
        log_string('%s' % df)




if __name__ == "__main__":
    
    train()
    LOG_FOUT.close()
