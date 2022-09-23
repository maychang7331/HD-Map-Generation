import os
import sys

import pandas as pd
import numpy as np
from cv2 import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from sklearn.cluster import DBSCAN

img = []

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def arrangeEOP2csv(csvfilelist, save_dir):
    
    X = []
    Y = []
    Z = []
    O = []
    P = []
    K = []
    img = []
    path = []
    camID = []

    for csv in csvfilelist:
        
        csv_data = pd.read_csv(csv, header=0)
        X.extend(csv_data['Origin (Easting[m]'].tolist())
        Y.extend(csv_data['Northing[m]'].tolist())
        Z.extend(csv_data['Height[m])'].tolist())
        O.extend(csv_data['Roll(X)[deg]'].tolist())
        P.extend(csv_data['Pitch(Y)[deg]'].tolist())
        K.extend(csv_data['Yaw(Z)[deg]'].tolist())
        # O.extend(csv_data['Omega[deg]'].tolist())
        # P.extend(csv_data['Phi[deg]'].tolist())
        # K.extend(csv_data['Kappa[deg]'].tolist())
        img.extend(csv_data['Filename'].tolist())
        path.extend([os.path.dirname(csv)]*len(csv_data))
        camID.extend([os.path.dirname(csv)[-1]]*len(csv_data))

    dic = {'X':X,
            'Y': Y,
            'Z': Z,
            'roll': O,
            'pitch': P,
            'yaw': K,
            'img': img,
            'path': path,
            'camID': camID}
    EOP = pd.DataFrame(dic, columns=['X', 'Y', 'Z', 'roll', 'pitch', 'yaw', 'img', 'path', 'camID'])
    # eop = pd.concat([pd.Series(v, name=n) for n, v in dic.items()], axis=1)
    EOP_fout = os.path.join(save_dir, 'EOP.csv')                              
    EOP.to_csv(EOP_fout, index=False)


def clusteringTargets(inputData, plot=True):
    
    clusters = DBSCAN(eps=0.45, min_samples=350).fit(inputData)
    
    # if np.count_nonzero(clusters.labels_ == -1) == len(clusters.labels_):
    #     print('targets within are noise !')
    #     return

    if plot:
        # Creating figure
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ='3d')

        # Creating plot
        X = inputData[:,0]
        Y = inputData[:,1]
        Z = inputData[:,2]
        ax.scatter3D(X,Y,Z, c=clusters.labels_, s=0.2)  # color須有實質顏色數值, 因此用c
        

        # ax.set_aspect('equal') 因為這個在z軸無法起作用，所以以下面方式取代
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')      

        # show plot
        plt.title("Clustering Result")
        plt.show()
    return clusters


def backprojection(eligible_img, points, iop):
    
    img = plt.imread(os.path.join(eligible_img[7], eligible_img[6]))
    cameraID = str(eligible_img[8])
    
    # 如果遇到camera 1(左後) 或 2(右後), 就跳過該影像（避免太多張拍到背面）
    # if cameraID == '1' or cameraID == '2':
    #     return []
    omega = eligible_img[3]*(np.pi/180)
    phi = eligible_img[4]*(np.pi/180)
    kappa = eligible_img[5]*(np.pi/180)

    # Ratation + Translation Homogeneous Matrix
    T = np.array(np.vstack([eligible_img[0], eligible_img[1], eligible_img[2]]))
    Rx = np.vstack([[1,0,0],[0,np.cos(omega),np.sin(omega)*(-1)],[0,np.sin(omega),np.cos(omega)]])
    Ry = np.vstack([[np.cos(phi),0,np.sin(phi)],[0,1,0],[np.sin(phi)*(-1),0,np.cos(phi)]])
    Rz = np.vstack([[np.cos(kappa),np.sin(kappa)*(-1),0],[np.sin(kappa),np.cos(kappa),0],[0,0,1]])
    R = Rz.dot(Ry).dot(Rx)
    RT = np.vstack([np.hstack([R,T]),[0,0,0,1]])
    
    # Mapping Frame to Camera Frame
    XYZ_M = np.vstack([points, np.ones((points.shape[1]))])
    XYZ_C = np.linalg.inv(RT).dot(XYZ_M)
    if any(XYZ_C[2,:] <= 0):   # 只要在相機的反側,就跳過該影像
        return []

    # reverse distortion correction
    fx = iop.loc['fx','Camera'+cameraID]
    fy = iop.loc['fx','Camera'+cameraID]
    Cx = iop.loc['Cx','Camera'+cameraID]
    Cy = iop.loc['Cy','Camera'+cameraID]
    k1 = iop.loc['k1','Camera'+cameraID]
    k2 = iop.loc['k2','Camera'+cameraID]
    k3 = iop.loc['k3','Camera'+cameraID]
    k4 = iop.loc['k4','Camera'+cameraID]
    k5 = iop.loc['k5','Camera'+cameraID]
    k6 = iop.loc['k6','Camera'+cameraID]
    p1 = iop.loc['p1','Camera'+cameraID]
    p2 = iop.loc['p2','Camera'+cameraID]
    
    # Camera Frame to Image Frame 需要乘這個 (單位也會從focal length -> pixel)
    K = np.vstack([[fx,0,Cx],[0,fy,Cy],[0,0,1]])

    xu = XYZ_C[0,:]/XYZ_C[2,:]  # 單位：meter -> focal length
    yu = XYZ_C[1,:]/XYZ_C[2,:]
    
    # 如果尚未加回透鏡畸變就不在影像範圍內,就跳過該影像（避免被修正量拉回相片內）
    xy_u = np.vstack([xu,yu,np.ones((points.shape[1]))])
    cr_u = K.dot(xy_u)
    if not all(np.logical_and(1<=cr_u[0,:], cr_u[0,:]<=img.shape[1])) or \
        not all(np.logical_and(1<=cr_u[1,:], cr_u[1,:]<=img.shape[0])):
        return []
    
    r = np.sqrt(np.square(xu)+np.square(yu))
    x_radial = xu * (k1*pow(r,2) + k2*pow(r,4) + k3*pow(r,6) + k4*pow(r,8))
    y_radial = yu * (k1*pow(r,2) + k2*pow(r,4) + k3*pow(r,6) + k4*pow(r,8))
    x_tangential = 2*p1*xu*yu + p2*(pow(r,2) + 2*pow(xu,2))
    y_tangential = 2*p2*xu*yu + p1*(pow(r,2) + 2*pow(yu,2))
    xd = xu + x_radial + x_tangential   # undistorted to distorted
    yd = yu + y_radial + y_tangential
    xy = np.vstack([xd,yd,np.ones((points.shape[1]))])
    cr = K.dot(xy)

    if all(np.logical_and(1<=cr[0,:], cr[0,:]<=img.shape[1])) and \
        all(np.logical_and(1<=cr[1,:], cr[1,:]<=img.shape[0])):
        return cr

    else:
        return []


def crop_img(cr, extend_pixel, fout):
    
    # c_max
    if np.ceil(max(cr[0])).astype(int) + extend_pixel < img.shape[1]:
        c_max = np.ceil(max(cr[0])).astype(int) + extend_pixel
    else:
        c_max = img.shape[1]

    # c_min
    if np.ceil(min(cr[0])).astype(int) - extend_pixel < 0:
        c_min = 0
    else:
        c_min = np.ceil(min(cr[0])).astype(int) - extend_pixel
                    
    # r_max
    if np.ceil(max(cr[1])).astype(int) + extend_pixel < img.shape[0]:
        r_max = np.ceil(max(cr[1])).astype(int) + extend_pixel
    else:
        r_max = img.shape[0]

    # r_min
    if np.ceil(min(cr[1])).astype(int) - extend_pixel < 0:
        r_min = 0
    else:
        r_min = np.ceil(min(cr[1])).astype(int) - extend_pixel
    
    # cr_max = np.ceil(np.max(cr[0:2], axis=1)).astype(int)
    # cr_min = np.ceil(np.min(cr[0:2], axis=1)).astype(int)

    crop_RGB = img[r_min:r_max, c_min:c_max]
    crop_BGR = cv2.cvtColor(crop_RGB, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fout, crop_BGR)


def projection_main(base_dir, pc_data_in, eop, iop, out_filename='proj_out'):

    """
    base_dir : the directory of where projection.py is
    pc_data_in : the directory of pointnet test output
    eop
    iop
    out_filename : 'proj_out' [default]
    
    """
    
    global img

    filename = os.path.basename(pc_data_in)
    print('\n%s ' %(filename))

    proj_out_dir = os.path.join(base_dir, out_filename)
    if not os.path.exists(proj_out_dir): 
        os.makedirs(proj_out_dir)

    # Load predict result
    predict_result = pd.read_csv(pc_data_in, delim_whitespace=True, header=None).values
    predict_result = np.unique(predict_result, axis=0)     # 用以刪除重複的點雲資料

    cond = (predict_result[:, -1] == 2)
    if np.sum(cond) == 0:
        print('no target within las!!')
        # continue #TODO
    target = predict_result[cond, :]
        
    # Clustering
    clusters = clusteringTargets(target[:, 0:4], plot=True)     # 放入intensity可有效濾除錯誤分類的點雲
    # if clusters == None:
    #     continue
    # else:
    num_of_cluster = len(np.unique(clusters.labels_))-1     # 拿掉noise (label = -1)

    geo_info_df = pd.DataFrame(columns=['geoLocation', 'bboxMin', 'bboxMax'])

    for i in range(num_of_cluster):
        cluster_cond = (clusters.labels_ == i)
        single_cluster = target[cluster_cond, :]
        XYZmax = np.amax(single_cluster[:,0:3], axis=0).reshape(1,3)
        XYZmin = np.amin(single_cluster[:,0:3], axis=0).reshape(1,3)
        cluster_center = (np.mean(np.array([XYZmax, XYZmin]), axis=0))
        
        # 將HD Map要的屬性紀錄在 geo_info 並輸出成csv
        geo_info_df.loc['cluster_'+str(i)] = \
            [" ".join("{:.3f}".format(i) for i in cluster_center[0])] + \
            [" ".join("{:.5f}".format(i) for i in XYZmin[0])] + \
            [" ".join("{:.5f}".format(i) for i in XYZmax[0])]

        # 找該cluster方圓10公尺內之相機外方位
        broad_cluster_center = np.broadcast_to(cluster_center, (len(eop), 3))
        dists = np.sqrt((np.square(eop[:, 0] - broad_cluster_center[:, 0]) 
                        + np.square(eop[:, 1] - broad_cluster_center[:, 1])).astype(float))
        dist_cond = (dists <= 20)
        eligible_img = eop[dist_cond, :]
        

        """
        # Draw buffer within 10
        ax = plt.axes(projection ='3d')
        ax.scatter3D(single_cluster[:, 0],single_cluster[:, 1],single_cluster[:, 2], c='blue', s=0.2)  # color須有實質顏色數值, 因此用c
        ax.scatter3D(eligible_img[:, 0],eligible_img[:, 1],eligible_img[:, 2], c='red', s=2)  # color須有實質顏色數值, 因此用c
        # Create cubic bounding box to simulate equal aspect ratio
        X = target[:, 0]
        Y = target[:, 1]
        Z = target[:, 2]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.show()
        """
        
        # 開資料夾以儲存 geo_info 和 crop出來的影像
        cluster_out_dir = os.path.join(proj_out_dir, os.path.splitext(filename)[0])
        if not os.path.exists(cluster_out_dir):
            os.makedirs(cluster_out_dir)      # mkdir with multiple depth
        img_out_dir = os.path.join(cluster_out_dir, 'cluster_' +str(i))
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)

        #### BACK PROJECTION
        for j in range(len(eligible_img)):
            
            # 先以cluster中心做反投影, 
            cr = backprojection(eligible_img[j], cluster_center.reshape(3,1), iop)
            if len(cr) != 0 :   # 若可成功反投影, 在將整個cluster丟去做反投影
                
                cr = backprojection(eligible_img[j], np.transpose(single_cluster[:,0:3]), iop)
                
                if len(cr) != 0 :   # 避免return false時沒東西畫
                    img = plt.imread(os.path.join(eligible_img[j,7], eligible_img[j,6]))
                    print(os.path.join(eligible_img[j,7], eligible_img[j,6]))
                    plt.imshow(img)
                    plt.scatter(cr[0,:], cr[1,:], c='#FFFF00', s=2)
                    # plt.show()

                    # crop image and save
                    crop_img(cr, extend_pixel=10, fout=img_out_dir + '/' + eligible_img[j,6])
            else:
                continue    
        
        print('proceeding... %d/%d (clusters found)' %(i+1,num_of_cluster))
        
    #### OUTPUT GEO INFORMATION OF EACH CLUSTER TO CSV
    geo_info_df.to_csv(os.path.join(cluster_out_dir, os.path.splitext(filename)[0]+'_cluster_geoinfo.csv'), index=True)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/chihyu/Desktop/thesis_2/projection
    sys.path.append(BASE_DIR)

    # DBSCAN tuning:
    # 9 10 11 15 : 0.45 650
    # 14 23 27 35 39 : 0.45 550
    PC_DATA_IN = '/home/chihyu/Desktop/thesis_2/pointnet/pointnet_out/log/shalun_ori_pt_23_pred.txt'
    IMG_DATA_DIR = '/home/chihyu/Desktop/Camera'

    # EOP
    while True:
        try:
            eop = pd.read_csv(os.path.join(BASE_DIR, 'EOP.csv'))
            eop = eop.to_numpy()        # turn dataframe to numpy
            break
        except IOError:
            _, csvfilelist = run_fast_scandir(IMG_DATA_DIR, [".csv"])
            arrangeEOP2csv(csvfilelist, save_dir=BASE_DIR)

    # IOP
    iop = pd.read_csv(os.path.join(BASE_DIR, 'IOP.csv'), delimiter="\t", index_col=0)
    projection_main(BASE_DIR, PC_DATA_IN, eop, iop, out_filename='proj_out')