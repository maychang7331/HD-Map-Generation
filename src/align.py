import os
import sys
import pandas as pd
from tensorflow.python.ops.gen_math_ops import sigmoid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/chihyu/Desktop/thesis_2
PROJ_DIR = os.path.join(BASE_DIR, 'projection')
CLAS_DIR = os.path.join(BASE_DIR, 'classification')
sys.path.insert(1, PROJ_DIR)
sys.path.insert(1, CLAS_DIR)

import projection
import test

projection_output_filename = 'proj_out'
classification_output_filename = 'clas_out'
FOUT = os.path.join(BASE_DIR, 'hdmap_out')
if not os.path.exists(FOUT): 
        os.makedirs(FOUT)

PNT_OUT_DIR = '/home/chihyu/Desktop/thesis/pointnet_out'
PROJ_OUT_DIR = os.path.join(PROJ_DIR, projection_output_filename)
CLAS_OUT_DIR = os.path.join(CLAS_DIR, classification_output_filename)
CLAS_LOG_DIR = os.path.join(CLAS_DIR, 'log')
MODEL_DIR = os.path.join(CLAS_LOG_DIR, 'kjo_net.validation.ckpt')

IMG_DATA_DIR = '/home/chihyu/Desktop/Camera'
'''
# EOP
while True:
    try:
        eop = pd.read_csv(os.path.join(PROJ_DIR, 'EOP.csv'))
        eop = eop.to_numpy()        # turn dataframe to numpy
        break
    except IOError:
        _, csvfilelist = projection.run_fast_scandir(IMG_DATA_DIR, [".csv"])
        projection.arrangeEOP2csv(csvfilelist, save_dir=PROJ_DIR)

# IOP
iop = pd.read_csv(os.path.join(PROJ_DIR, 'IOP.csv'), delimiter="\t", index_col=0)

predList = []
for pointnet_fout in os.listdir(PNT_OUT_DIR):
    if pointnet_fout.endswith("_pred.txt"):
        predList.append(os.path.splitext(pointnet_fout)[0])   
        pc_data_in = os.path.join(PNT_OUT_DIR, pointnet_fout)
        projection.projection_main(PROJ_DIR, pc_data_in, eop, iop, projection_output_filename)
predList = sorted(predList)
print('Back projection process done !!')

test.test_classification(CLAS_DIR, MODEL_DIR, PROJ_OUT_DIR, classification_output_filename)
'''

_, proj_csvfilelist = projection.run_fast_scandir(PROJ_OUT_DIR, [".csv"])
_, clas_csvfilelist = projection.run_fast_scandir(CLAS_OUT_DIR, [".csv"])

proj_csvfilelist = sorted(proj_csvfilelist)
clas_csvfilelist = sorted(clas_csvfilelist)

assert(len(proj_csvfilelist) == len(clas_csvfilelist))," The number of csv found in proj_out and clas_out does not match"

for i in range(len(proj_csvfilelist)):
    class_df = pd.read_csv(clas_csvfilelist[i], index_col=0)    # 將第一列讀成index
    geoinfo_df = pd.read_csv(proj_csvfilelist[i], index_col=0)    # 將第一列讀成index
    
    cond = class_df['Ratio'] > 5.0
    if cond.sum() != 0:
        
        sign_df = pd.DataFrame(columns=['code'])
        
        # 根據每列最大值取得column name 
        # ref:https://stackoverflow.com/questions/29919306/find-the-column-name-which-has-the-maximum-value-for-each-row
        tmp_df = class_df.loc[cond]
        columns = tmp_df.columns.values.tolist()[1:]            # 移除BACK 欄位  
        sign_df['code'] = tmp_df[columns].idxmax(axis=1)
        sign_df = pd.concat([sign_df, geoinfo_df[cond]], axis=1)
        sign_df.index = range(len(sign_df.index))                         # 重設 index, 使不為cluster_XX
        sign_df.to_csv(os.path.join(FOUT, predList[i]+'_sign.csv'), index=True)
    else:
        continue
print('End !')