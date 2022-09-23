import matplotlib.pyplot as plt
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logdso8(kp0.6 valid_rnd)')
# LOG_DIR = '/home/chihyu/Desktop'
epoch = []
epoch_mean_loss = []
epoch_accuracy = []
val_mean_loss = []
val_accuracy = []
val_avg_class_acc = []
with open(LOG_DIR +'/log_train.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '**** EPOCH ' in line.strip():
            epoch.append(int(line.split()[-2]))
        elif '(epoch) mean loss' in line.strip():
            epoch_mean_loss.append(float(line.split()[-1]))
        elif '(epoch) accuracy' in line.strip():
            epoch_accuracy.append(float(line.split()[-1]))
        elif 'valid mean loss' in line.strip():
            val_mean_loss.append(float(line.split()[-1]))
        elif 'valid accuracy' in line.strip():
            val_accuracy.append(float(line.split()[-1]))
        elif 'valid avg class acc' in line.strip():
            val_avg_class_acc.append(float(line.split()[-1]))
    f.close()

fig, (loss, acc) = plt.subplots(1, 2, figsize=(14,6))    # 1 row 2 column
fig.suptitle('Learning curve (valid: random 20% cuboids)\n \
            [lr=0.001, step=300000, rate=0.5]', fontsize=12)
line1, = loss.plot(epoch, epoch_mean_loss, color='tab:blue', label='train')
line2, = loss.plot(epoch, val_mean_loss, color='tab:orange', label='valid')
loss.legend(handles = [line1, line2], loc='upper left')
loss.set_title('Model Loss')

line3, = acc.plot(epoch, epoch_accuracy, color='tab:blue', label='train')
line4, = acc.plot(epoch, val_accuracy, color='tab:orange', label='valid')
line5, = acc.plot(epoch, val_avg_class_acc, color='tab:red', label='valid avg class')
acc.legend(handles = [line3, line4, line5], loc='lower right')
acc.set_title('Model Accuracy')
# plt.show()
fig.savefig(LOG_DIR + '/graph.png')

# line1, = plt.plot(epoch, epoch_mean_loss, color='tab:blue', linestyle='--', label='train mean loss')
# line2, = plt.plot(epoch, epoch_accuracy, color='tab:blue', label='train accuracy')
# line3, = plt.plot(epoch, val_mean_loss, color='tab:orange', linestyle='--', label='valid mean loss')
# line4, = plt.plot(epoch, val_accuracy, color='tab:orange', label='valid accuracy')
# line5, = plt.plot(epoch, val_avg_class_acc, color='firebrick',  label='valid accuracy')
# plt.legend(handles = [line1, line2, line3, line4, line5], loc='center right')
# plt.show()