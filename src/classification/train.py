import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from cv2 import cv2


# Machine Learning
# from sklearn.utils import shuffle  # Used numpy's shuffle instead.
import tensorflow as tf
# TODO
from model import kjo_net
from model_2 import GoogLeNet
from sklearn.model_selection import train_test_split


# Project-Specific.
# Includes custom functions for
# image processing, data manipulation, and dataset visualization.
from classification_utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/chihyu/Desktop/thesis_2/classification
sys.path.append(BASE_DIR)

LOG_DIR = os.path.join(BASE_DIR, 'log')    # log(GoogleNet)_railway_aug
if not os.path.exists(LOG_DIR): 
    os.makedirs(LOG_DIR)
FIG_DIR = os.path.join(LOG_DIR, 'fig')
if not os.path.exists(FIG_DIR): 
    os.makedirs(FIG_DIR)

DATA_DIR = os.path.join(BASE_DIR, "signData_railway")           # folder with all the class folders
labelFile = os.path.join(BASE_DIR, 'labels_railway.csv')        # file with all names of classes

testRatio = 0.2         # if 1000 images split will 200 for testing
validationRatio = 0.2   # if 1000 images 20% of remaining 800 will be 160 for validation
imageDimesions = (32,32,3)


##### IMPORTING IMAGES
images = []
classID = []
classList = os.listdir(DATA_DIR)
numOfClasses = len(classList)       # 43
for i in range(numOfClasses):
    classList[i] = int(classList[i])
classList = sorted(classList)

for i in range(numOfClasses):
    PIC_DIR = os.path.join(DATA_DIR, str(classList[i]))
    for pic in os.listdir(PIC_DIR):
        img_bgr = cv2.imread(os.path.join(PIC_DIR,pic))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (32, 32))
        images.append(img_rgb_resized)
        classID.append(classList[i])    
    print(str(classList[i]), end=" ", flush=True)
print(" ")
images = np.array(images)
classID = np.array(classID)


##### SPLIT DATA
X_train, X_validation, y_train, y_validation = train_test_split(images, classID, test_size=testRatio, stratify=classID)
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio, stratify=y_train)


##### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
print("DATA SHAPES -----------")
print("Train Shapes = ", X_train.shape, y_train.shape)                  # Train(22271, 32, 32, 3) (22271,)
print("Validation Shapes = ", X_validation.shape, y_validation.shape)   # Validation(5568, 32, 32, 3) (5568,)
# print("Test Shapes = ", X_test.shape, y_test.shape)                     # Test(6960, 32, 32, 3) (6960,)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
# assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(imageDimesions))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimesions))," The dimesionas of the Validation images are wrong "
# assert(X_test.shape[1:]==(imageDimesions))," The dimesionas of the Test images are wrong"


##### READ LABEL CSV FILE
lbl_df = pd.read_csv(labelFile)
lbl_df = lbl_df.set_index('Id')
lbl_dict = lbl_df['Code'].to_dict()
print("label discription shape", lbl_df.shape)


##### PLOT REPRESENTATIVE IMAGE OF EACH CLASS
X_all, y_all = combine(X_train, y_train, X_validation, y_validation)
images_by_label = sort_by_class(X_all, y_all, classList)

plot_representative_images(images_by_label, class_cnt=numOfClasses, english_labels=lbl_dict,
                            method='median', image_cnt=None)
# plt.savefig(os.path.join(FIG_DIR, 'class_representative_images.png'))


##### PLOT HISTOGRAM OF CLASS IN EACH DATASET
class_histograms(
    y_train, 
    y_validation,
    # y_test,
    class_cnt=numOfClasses)
# plt.savefig(os.path.join(FIG_DIR, 'class_histograms.png'))


##### PLOT EXAMPLES OF IMAGE AUGMENTATION
demo_image = X_train[10, :]
test_randomly_perturb(
                    demo_image,
                    brightness_radius=0.3,
                    rotation_radius=30.0,
                    translation_radius=10,
                    shear_radius=10)
plt.show()
# plt.savefig(os.path.join(FIG_DIR, 'random_perturb.png'))

##### PLOT HISTOGRAM OF CLASS IN EACH DATASET AFTER AUGUMENATION
DATAPOINTS_PER_CLASS_FOR_TRAINING = 150  # Default 300.
X_train_balanced, y_train_balanced = balance(
    X_train, y_train, classList, datapoint_cnt_per_class=DATAPOINTS_PER_CLASS_FOR_TRAINING, perturb=True)

X_validate_balanced, y_validate_balanced = balance(
    X_validation, y_validation, classList, datapoint_cnt_per_class=None, perturb=True)
    
# X_test_balanced, y_test_balanced = balance(
# X_test, y_test, classList, datapoint_cnt_per_class=None, perturb=False)

class_histograms(
    y_train_balanced,
    y_validate_balanced,
    # y_test_balanced,
    class_cnt=numOfClasses,
    suptitle_prefix='Balanced')
# plt.savefig(os.path.join(LOG_DIR, 'fig/class_histograms-balanced.png'))


##### PLOT EXAMPLES OF EQUALIZATION IN BRIGHTNESS
test_histogram_equalize_brightness(X_train[10, :])
# plt.savefig(os.path.join(FIG_DIR, 'histogram_brightness_equalization.png'))
plt.show()

##### PREPROCESS IMAGES BEFORE PUT INTO CNN
X_train_balanced_in = preprocess(X_train_balanced)
X_train_in, y_train = X_train_balanced_in, y_train_balanced

# X_validate_balanced_in = preprocess(X_validate_balanced)
X_validate_in = preprocess(X_validation)

# X_test_balanced_in = preprocess(X_test_balanced)
# X_test_in = preprocess(X_test)

"""
################################# model LeNet #####################################
# Hyperparameters.
LEARNING_RATE = 0.001       # Default 0.001
EPOCH_CNT = 30              # Number of times to run over training data. Default 30.
BATCH_SIZE = 24             # Datapoints to consider per backprop pass. Default 256.
KEEP_PROBABILITY = 0.4      # For dropout regularization.  Try 0.5, 0.6, 0.7, 1.0

# Input images and output labels.
X_in = tf.placeholder(tf.float32, (None, 32, 32, 1))    # None => arbitrary batch size.
y = tf.placeholder(tf.int32, (None))                    # None => arbitrary batch size.
y_one_hot = tf.one_hot(y, numOfClasses)
keep_probability = tf.placeholder(tf.float32)

logits = kjo_net(X_in, keep_probability, numOfClasses)
"""

################################# model_googleNet  #####################################
# Hyperparameters.
LEARNING_RATE = 4e-4       # Default 0.001
EPOCH_CNT = 30              # Number of times to run over training data. Default 30.
BATCH_SIZE = 24             # Datapoints to consider per backprop pass. Default 24.
KEEP_PROBABILITY = 0.5

X_in = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, numOfClasses)
keep_probability = tf.placeholder_with_default(1.0, shape=())

logits = GoogLeNet(X_in, keep_probability, numOfClasses)

###################################################################################33
# Train Method
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# To save parameters for re-use.
saver = tf.train.Saver()  # Must be initialized after tf symbolic variables.
validation_save_filename = os.path.join(LOG_DIR,'kjo_net.validation.ckpt')

def compute_accuracy(session, X_in_, y_):
    '''Compute model accuracy.
    
    ::WARNING:: The trailing underscores on the argument variable names
    are to prevent collision with global tensoflow variables X_in and y.
    
    Args:
        session: tf.session.
        X_in_: np.array, preprocessed inputs.
        y_: np.array, outputs corresponding to X_in.
        
    Returns:
        float, accuracy of model applied to (X_in, y).
    '''
    
    #session = tf.get_default_session()
    example_cnt = len(X_in_)
    total_accuracy = 0
    for offset in range(0, example_cnt, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_X_in, batch_y = X_in_[offset:end], y_[offset:end]
        accuracy = session.run(accuracy_operation,
                                feed_dict={X_in: batch_X_in, y: batch_y, keep_probability: 1.0})
        total_accuracy += (accuracy * len(batch_X_in))
        
    return total_accuracy / example_cnt

def compute_loss(session, X_in_, y_):
    example_cnt = len(X_in_)
    num_batches = X_in_.shape[0] // BATCH_SIZE
    loss_sum = 0
    for offset in range(0, example_cnt, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_X_in, batch_y = X_in_[offset:end], y_[offset:end]
        loss_val = session.run(loss_operation,
                                feed_dict={X_in: batch_X_in, y: batch_y, keep_probability: KEEP_PROBABILITY})
        loss_sum += loss_val
    return loss_sum / float(num_batches)


start_time = time.time()
accuracies_train = []
accuracies_validate = []
accuracies_validate_balanced = []

loss_train = []
loss_validate = []
loss_validate_balanced = []

with tf.Session() as session:
    training_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  session.graph)    # TODO
    session.run(tf.global_variables_initializer())
    example_cnt = len(X_train_in)
    print('Training...\n')
    for i in range(EPOCH_CNT):
        
        X_train_in, y_train = shuffle(X_train_in, y_train)

        for offset in range(0, example_cnt, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_X_in, batch_y = X_train_in[offset:end], y_train[offset:end]
            
            session.run(training_operation,
                        feed_dict={X_in: batch_X_in, y: batch_y, keep_probability: KEEP_PROBABILITY})

        accuracy_train = compute_accuracy(session, X_train_in, y_train)
        accuracies_train.append(accuracy_train)
        
        accuracy_validate = compute_accuracy(session, X_validate_in, y_validation)
        accuracies_validate.append(accuracy_validate)

        # accuracy_validate_balanced = compute_accuracy(session, X_validate_balanced_in, y_validate_balanced)
        # accuracies_validate_balanced.append(accuracy_validate_balanced)

        loss_t = compute_loss(session, X_train_in, y_train)
        loss_train.append(loss_t)
        
        loss_v = compute_loss(session, X_validate_in, y_validation)
        loss_validate.append(loss_v)

        # loss_v_balanced = compute_loss(session, X_validate_balanced_in, y_validate_balanced)
        # loss_validate_balanced.append(loss_v_balanced)
        
        print('**** EPOCH {} ****'.format(i+1))
        print('Training Accuracy = {:.3f}'.format(accuracy_train))
        print('Validation Accuracy = {:.3f}'.format(accuracy_validate))
        # print('Balanced Validation Accuracy = {:.3f}'.format(accuracy_validate_balanced))
        print('---')
        print('Training Loss = {:.3f}'.format(loss_t))
        print('Validation Loss = {:.3f}'.format(loss_v))
        # print('Balanced Validation Loss = {:.3f}'.format(loss_v_balanced))
        print()

    saver.save(session, validation_save_filename)
    print("Model saved")

dt_s = time.time() - start_time
dt_m = dt_s / 60.0
dt_h = dt_s / 3600.0
print('Wallclock time elapsed: {:.3f} s = {:.3f} m = {:.3f} h.'.format(dt_s, dt_m, dt_h))

# Plot accuracies vs epoch.
plt.figure()
plt.plot(np.arange(EPOCH_CNT), accuracies_train, label='training accuracy', zorder=0, linewidth=3)
plt.plot(np.arange(EPOCH_CNT), accuracies_validate, label='validation accuracy', zorder=0, linewidth=3)
# plt.plot(np.arange(EPOCH_CNT), accuracies_validate_balanced, label='balanced validation accuracy', zorder=0, linewidth=3)

plt.title('Accuracies vs Epoch', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
legend = plt.legend(numpoints=1)
plt.setp(legend.get_texts(), fontsize='14', fontweight='bold')
ax = plt.gca()
#plt.axis('equal') # Defective so use set_aspect instead?
#ax.set_aspect('equal', adjustable='box')
ax.margins(0.1)
#ax.set_yscale('log')
#ax.autoscale(tight=True)
#plt.xlim((0, 2000.0))
#plt.ylim((0, 1000.0))
plt.grid(True)
plt.show()
# plt.savefig(os.path.join(FIG_DIR, 'accuracies_vs_epoch.png'))

# Plot loss vs epoch.
plt.figure()
plt.plot(np.arange(EPOCH_CNT), loss_train, label='training loss', zorder=0, linewidth=3)
plt.plot(np.arange(EPOCH_CNT), loss_validate, label='validation loss', zorder=0, linewidth=3)
# plt.plot(np.arange(EPOCH_CNT), loss_validate_balanced, label='balanced validation loss', zorder=0, linewidth=3)

plt.title('Loss vs Epoch', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
legend = plt.legend(numpoints=1)
plt.setp(legend.get_texts(), fontsize='14', fontweight='bold')
ax = plt.gca()
ax.margins(0.1)
plt.grid(True)
plt.show()
# plt.savefig(os.path.join(FIG_DIR, 'loss_vs_epoch.png'))