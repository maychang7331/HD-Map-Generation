import os
import numpy as np
import pandas as pd
import provider
# from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.basename(BASE_DIR)
ALL_FILES = provider.getDataFiles(os.path.join(ROOT_DIR,'data/shalun_h5/dso8rmpole/all_h5files.txt'))
block_filelist = [line.rstrip() for line in open(os.path.join(ROOT_DIR,'data/shalun_h5/dso8rmpole/block_filelist.txt'))]
# ALL_FILES = provider.getDataFiles('/home/chihyu/Desktop/dso8newmerge/all_h5files.txt')
# block_filelist = [line.rstrip() for line in open('/home/chihyu/Desktop/dso8newmerge/block_filelist.txt')]

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
(unique, counts) = np.unique(label_batches, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)
print(data_batches.shape)
print(label_batches.shape)

# blocks_with_sign = np.empty((len(label_batches), 1))
# for i in range(len(label_batches)):
#     if np.sum(label_batches[i, :]==2) != 0:
#         blocks_with_sign[i] = True
#     else:
#         blocks_with_sign[i] = False
# print(np.sum(blocks_with_sign))
# sign_blocks = data_batches[blocks_with_sign, :]
# print(sign_blocks.shape)

print("END !!!")

""" <test confusion matrix>
tic = datetime.now()
NUM_CLASSES = 5
LOG_DIR = '/home/chihyu/Desktop'
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

diagonal = np.zeros([1, NUM_CLASSES])
gt_total = np.zeros([NUM_CLASSES, 1])
pred_total = np.zeros([1, NUM_CLASSES])
precision =  np.zeros([1, NUM_CLASSES])
recall = np.zeros([1, NUM_CLASSES])
f1_score = np.zeros([1, NUM_CLASSES])

confusion_matrix = np.ones((5, 5))
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
confusion_matrix = np.concatenate((confusion_matrix, gt_total[:, None]), axis=1)      # 加到最後一行
confusion_matrix = np.concatenate((confusion_matrix, pred_total.T), axis=0)    # 加到最後一列
confusion_matrix = np.concatenate((confusion_matrix, precision.T), axis=0)
confusion_matrix = np.concatenate((confusion_matrix, recall.T), axis=0)
confusion_matrix = np.concatenate((confusion_matrix, f1_score.T), axis=0)

df = pd.DataFrame(confusion_matrix, 
                    columns=['other', 'traffic island', 'sign', 'signal', 'pole', 'Total'],
                    index=['other', 'traffic island', 'sign', 'signal', 'pole', 'Total', 'Precision', 'Recall', 'F1 Score'])
df.T.round(decimals=dict(zip(df.index, [0, 0, 0, 0, 0, 0, 2, 2, 2]))).T
df = df.astype(object)
# print(df)
log_string('%s' % df)

toc = datetime.now()
print('Elapsed time: %f seconds' % (toc-tic).total_seconds())
"""

""" <test predict file out>
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
district_data_filelist = 'meta/pt9_16_data_label.txt'
DISTRICT_PATH_LIST = [os.path.join(BASE_DIR,line.rstrip()) for line in open(district_data_filelist)]
DUMP_DIR = 'log/dump'

for district_path in DISTRICT_PATH_LIST:
    print(os.path.basename(district_path))
    print(os.path.basename(district_path)[:-4])
    out_data_label_filename = os.path.basename(district_path)[:-4] + '_pred.txt'
    out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
    out_gt_label_filename = os.path.basename(district_path)[:-4] + '_gt.txt'
    out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
    print(district_path, out_data_label_filename)
"""