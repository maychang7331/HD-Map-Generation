import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import tensorflow as tf

import classification_utils
import model
import model_2


def test_classification(base_dir, model_dir, proj_data_in, out_filename='clas_out'):

    imageDimesions = (32,32,3)

    #### CREATE OUTPUT DIRECTORY FOR CLASSIFICATION
    clas_out_dir = os.path.join(base_dir, out_filename)
    if not os.path.exists(clas_out_dir): 
        os.makedirs(clas_out_dir)

    ##### READ LABEL CSV FILE
    labelFile = os.path.join(base_dir, 'labels_railway.csv')    # labels.csv
    lbl_df = pd.read_csv(labelFile)
    lbl_df = lbl_df.set_index('Id')
    lbl_dict = lbl_df['Code'].to_dict()
    num_of_classes = lbl_df.shape[0]

    """
    ############################## model LeNet ###################################
    X_in = tf.placeholder(tf.float32, (None, 32, 32, 1))  # None => arbitrary batch size.
    keep_probability = tf.placeholder(tf.float32)
    logits = model.kjo_net(X_in, keep_probability, num_of_classes)
    saver = saver = tf.train.Saver()
    """

    ############################## model_2 GoogleNet ###################################
    X_in = tf.placeholder(tf.float32, (None, 32, 32, 3))  # None => arbitrary batch size.
    keep_probability = tf.placeholder_with_default(1.0, shape=())
    logits = model_2.GoogLeNet(X_in, keep_probability, num_of_classes)
    saver = saver = tf.train.Saver()
    
    
    with tf.Session() as session:
        saver.restore(session, model_dir) 
        

        # Loop predict files in proj_out
        predList = [predFolder for predFolder in os.listdir(proj_data_in) \
                            if os.path.isdir(os.path.join(proj_data_in, predFolder))]
        predList = sorted(predList)  
        
        for predFile in predList:

            print('Classifying clusters in %s' % predFile)
            predFile_dir = os.path.join(proj_data_in, predFile)
            clusterList = [clusterFolder for clusterFolder in os.listdir(predFile_dir) \
                            if os.path.isdir(os.path.join(predFile_dir, clusterFolder))]
            clusterList = sorted(clusterList)
            
            # Create array for voting mechanism
            voting_arr = np.zeros([len(clusterList), num_of_classes])

            # Loop clusters in each predicted file
            for idx_c, clusterFile in enumerate(clusterList):
                
                clusterFile_dir = os.path.join(predFile_dir,clusterFile)
                imgList = os.listdir(clusterFile_dir)

                # Append cropped images in each cluster for predicting
                X_new = np.zeros((len(imgList), *imageDimesions), dtype=np.uint8)
                for idx_i, img in enumerate(imgList):
                    img_bgr = cv2.imread(os.path.join(clusterFile_dir, img))
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_rgb_resized = cv2.resize(img_rgb, (32, 32))
                    X_new[idx_i, :] = img_rgb_resized
                classification_utils.plot_images(X_new)
                # plt.show()

                X_new_in = classification_utils.preprocess(X_new)     
                top_k = session.run(tf.nn.top_k(tf.nn.softmax(logits), k=5),
                                    feed_dict={X_in: X_new_in, keep_probability: 1.0})
                
                # Plot softmax probabilities.
                probabilities, classes = top_k
                ixs = np.arange(5)
                probability_barh_width = 14.5
                probability_barh_height = 3
                
                for i, image in enumerate(X_new):
                    print(imgList[i])
                    print('\nImage %g\n  Predicted class %g: %s' % (i, top_k[1][i, 0], lbl_dict[top_k[1][i, 0]]))

                    # 對於照片有80%的成功機率才納入voting array計算
                    if top_k[0][i, 0] > 0.80:    # top_k[0][i, 0] == probabilities[i, 0]
                        voting_arr[idx_c, top_k[1][i, 0]] +=1

                    fig, axes = plt.subplots(1, 2, figsize=(probability_barh_width, probability_barh_height))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    
                    plt.subplot(1, 2, 2)
                    ax = plt.gca()
                    plt.barh(ixs, probabilities[i], height=0.5, align='center')
                    y_tick_labels = []    
                    for j in ixs: 
                        label = str(classes[i][j]) + ': ' + \
                        lbl_dict[classes[i][j]]
                        y_tick_labels.append(classification_utils.truncate_string(label, ubnd=25))
                    plt.yticks( ixs, y_tick_labels)
                    plt.title('Top 5 Class Probabilities')
                    # plt.xlabel('Probability')
                    # ax = plt.gca(); ax.set_xscale('log') 
                    ax.invert_yaxis()
                    for i, value in enumerate(probabilities[i]):
                        ax.text(value + 0.03, i + .03, str(value), color='blue', fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xlim(0.0, 1.0)
                    plt.show()
                    # plt.savefig('./fig/new_image-%03g.top_softmax_probabilities.png' % i)
                
                print('%d/%d done' %(idx_c+1, len(clusterList)))
            print(" ")


            voting_mask = voting_arr[:, 1:]                                     # 把 BACK(標誌背面) 欄位拿掉
            voting_sort = np.zeros((len(clusterList), voting_mask.shape[1] - 1))
            for i in range(len(voting_mask)):
                voting_sort[i, :] = sorted(voting_mask[i])[0:-1]                # 由小道大排序各類別的照片數, 並拿掉最大值(最後一個)
            
            voting_sort = np.array(voting_sort)
            voting_sort[voting_sort == 0] = np.NAN                              # 把 0 用 nan 取代以利計算 Ratio = 最大的照片數/(其餘照片數的平均值)
            Modified_SNR = (np.nanmax(voting_mask,axis=1) / np.nanmean(voting_sort, axis=1)).reshape((len(clusterList), 1))
            voting_out = np.concatenate((voting_arr, Modified_SNR), axis=1)     # 合併至原 voting array 的最後一欄

            # 轉成 dataframe 格式並輸出成 csv
            col_names = [value for key, value in lbl_dict.items()]
            col_names.append('Modified_SNR')
            voting_df = pd.DataFrame(voting_out, columns=col_names, index=clusterList)
            voting_df = voting_df.round(decimals=1)
            voting_df.to_csv(os.path.join(clas_out_dir, predFile)+'_cluster_classification.csv', index=True)
            print(voting_df)
        print('Classification process done !!')

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/chihyu/Desktop/thesis_2/classification
    sys.path.append(BASE_DIR)

    # Import trained model from log directory of classification
    LOG_DIR = os.path.join(BASE_DIR, 'log')  # log(GoogleNet)_railway_aug
    MODEL_DIR = os.path.join(LOG_DIR, 'kjo_net.validation.ckpt')

    # Get data from projection output
    ROOT_DIR = os.path.dirname(BASE_DIR)
    PROJ_DIR = os.path.join(ROOT_DIR, 'projection')
    PROJ_DATA_IN = os.path.join(PROJ_DIR, 'proj_out')
    
    test_classification(BASE_DIR, MODEL_DIR, PROJ_DATA_IN, out_filename='clas_out')   # clas_rmpole(valid_rnd)_buffer20_railway0726
