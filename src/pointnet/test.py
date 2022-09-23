import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)
from model import *
import outdoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
# parser.add_argument('--model_path', required=True, help='model checkpoint file path')   # 輸入model
# parser.add_argument('--dump_dir', required=True, help='dump folder path')   # 輸出目錄
# parser.add_argument('--output_filelist', required=True, help='TXT filename, filelist, each line is an output for a district')   # 輸出文件列表
# parser.add_argument('--district_data_filelist', required=True, help='TXT filename, filelist, each line is a test district data label file.')    # 輸入：測試文件列表
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(BASE_DIR,'logrmpole(kp0.6 valid_rnd)/model.ckpt')          # FLAGS.model_path #################################################
GPU_INDEX = FLAGS.gpu
DUMP_DIR = os.path.join(BASE_DIR, 'pointnet_out/logrmpole(kp0.6 valid_rnd)_givemeaccuracy')          # FLAGS.dump_dir ###################################################
if not os.path.exists(DUMP_DIR): 
    os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
DISTRICT_PATH_LIST = [os.path.join(BASE_DIR,line.rstrip()) for line in open(os.path.join(BASE_DIR,'meta/ori_pt9_16.txt'))]   # FLAGS.district_data_filelist ###################################

NUM_CLASSES = 5
NUM_FEATURES = 4

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    total_correct = 0
    total_seen = 0
    fout_out_filelist = open(DUMP_DIR + '_output_filelist.txt', 'w')        # FLAGS.output_filelist ###################################################
    for district_path in DISTRICT_PATH_LIST:
        out_data_label_filename = os.path.basename(district_path)[:-4] + '_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = os.path.basename(district_path)[:-4] + '_gt.txt'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
        log_string('%s %s' %(district_path, out_data_label_filename))
        a, b = eval_one_epoch(sess, ops, district_path, out_data_label_filename, out_gt_label_filename)
        if a == -1:
            continue
        total_correct += a
        total_seen += b
        fout_out_filelist.write(out_data_label_filename+'\n')
    fout_out_filelist.close()
    log_string('all district eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, district_path, out_data_label_filename, out_gt_label_filename):
    error_cnt = 0
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

    if FLAGS.visu:
        # write .obj
        fout = open(os.path.join(DUMP_DIR, os.path.basename(district_path)[:-4]+'_pred.obj'), 'w')
        fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(district_path)[:-4]+'_gt.obj'), 'w')
    # write .txt
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')
    
    current_data, current_label = outdoor3d_util.district2blocks_wapper_normalized(district_path, NUM_POINT, block_size=2.5, stride=1.0)
    if len(current_data)==0:
        return -1, -1
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)


    # Get district dimension..
    # data_label = np.load(district_path)
    # data = data_label[:,0:4]
    # max_district_x = max(data[:,0])
    # max_district_y = max(data[:,1])
    # max_district_z = max(data[:,2])
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        # current_data: X Y X" Y" Z I' R' G' B' (N X 9)，訓練只放 X" Y" Z I' (2:6)
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, 2:6],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN 
        else:
            pred_label = np.argmax(pred_val, 2) # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx+b, :, :]       # X Y X" Y" Z I' R' G' B' (N X 9)
            pts[:, 5] *= 65535
            # pts[:, 6:9] *= 255
            # pts[:, 6:9] = np.round(pts[:, 6:9]).astype(int)
            l = current_label[start_idx+b,:]
            # pts[:,6] *= max_district_x
            # pts[:,7] *= max_district_y
            # pts[:,8] *= max_district_z
            # pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                color = outdoor3d_util.g_label2color[pred[i]]
                color_gt = outdoor3d_util.g_label2color[current_label[start_idx+b, i]]
                if FLAGS.visu:
                    # write .obj
                    fout.write('v %f %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,4], pts[i,5], color[0], color[1], color[2]))
                    fout_gt.write('v %f %f %f %f %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,4], pts[i,5], color_gt[0], color_gt[1], color_gt[2]))
                # write .txt
                fout_data_label.write('%f %f %f %f %d %d %d %f %d\n' % (pts[i,0], pts[i,1], pts[i,4], pts[i,5], color[0], color[1], color[2], pred_val[b,i,pred[i]], pred[i]))
                fout_gt_label.write('%f %f %f %f %d %d %d %d\n' % (pts[i,0], pts[i,1], pts[i,4], pts[i,5], color_gt[0], color_gt[1], color_gt[2], l[i]))
        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)
                confusion_matrix[l, pred_label[i-start_idx, j]] +=1

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    
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
    log_string('%s' % df)


    fout_data_label.close()
    fout_gt_label.close()
    if FLAGS.visu:
        fout.close()
        fout_gt.close()
    return total_correct, total_seen


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
