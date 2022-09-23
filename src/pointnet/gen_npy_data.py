import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/chihyu/Desktop/thesis
sys.path.append(BASE_DIR)
import outdoor3d_util

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/shalun_txt_ori_dir.txt'))]
folder_list = os.path.normpath(anno_paths[0]).split(os.path.sep)

output_dir = os.path.join(BASE_DIR, 'data/shalun_npy/' + folder_list[-2])

if not os.path.exists(output_dir):
    os.makedirs(output_dir)      # mkdir with multiple depth

for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3].split('_')[0] +'_' + elements[-2] + '_' + elements[-1].split('.')[0] + '.npy'
        outdoor3d_util.collect_point_label(anno_path, os.path.join(output_dir, out_filename), 'numpy')
        # X Y X' Y' Z I R G B L
    except:
        print(anno_path, 'ERROR!!')

# .npy 檔案輸出成txt檔，存在meta資料夾
filelist = []
for file in os.listdir(output_dir):
    if file.endswith('.npy'):
        filelist.append(file)
filelist.sort(key=lambda f: int(f.split('.')[0].split('_')[-1]))    # 按數字大小排序

npyfilename = os.path.join(BASE_DIR, 'meta/' +output_dir.split('/')[-2] +'_' +output_dir.split('/')[-1] +'.txt')
with open(npyfilename, 'w') as f:
    for item in filelist:
        f.write('%s\n' % item)
    f.close()