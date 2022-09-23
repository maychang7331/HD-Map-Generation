import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

PC_DIR = '/home/chihyu/Desktop/shalun/testing_ts'
FOUT = '/home/chihyu/Desktop/thesis_2/ts_geoinfo.txt'
fout = open(FOUT, 'w')

for file in os.listdir(PC_DIR):

    if file.endswith(".txt"):
        ts_pc_dir = os.path.join(PC_DIR, file)

        ts_pc = np.loadtxt(ts_pc_dir)          # XYZIRGBL

        # inputData = ts_pc[:, 0:3]
        # clusters = DBSCAN(eps=0.10, min_samples=0).fit(inputData)

        # # Creating figure
        # fig = plt.figure(figsize = (10, 7))
        # ax = plt.axes(projection ='3d')

        # # Creating plot
        # X = inputData[:,0]
        # Y = inputData[:,1]
        # Z = inputData[:,2]
        # ax.scatter3D(X,Y,Z, c=clusters.labels_, s=0.2)  # color須有實質顏色數值, 因此用c
        # plt.show()

        # num_of_cluster = len(np.unique(clusters.labels_))-1     # 拿掉noise (label = -1)

        # if num_of_cluster == 1:
        #     print(ts_pc_dir)
        #     cluster_cond = (clusters.labels_ != -1)
        #     cluster_without_noise = inputData[cluster_cond, :]

        XYZmax = np.amax(ts_pc[:,0:3], axis=0).reshape(1,3)
        XYZmin = np.amin(ts_pc[:,0:3], axis=0).reshape(1,3)
        cluster_center = (np.mean(np.array([XYZmax, XYZmin]), axis=0))

        fout.write('%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n' % 
                                                (file, 
                                                    cluster_center[0,0], cluster_center[0,1], cluster_center[0,2],
                                                    XYZmin[0,0], XYZmin[0,1], XYZmin[0,2],
                                                    XYZmax[0,0], XYZmax[0,1], XYZmax[0,2]))
