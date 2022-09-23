import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cv2 import cv2  # Image processing.
from augmentation_utils import *


# ----- Image Processing -----
def rgb2gray(image, add_degenerate_dimension=False):
    '''Convert RGB image to grayscale.
    Args:
        image: numpy array, RGB image.
        add_degenerate_dimension: bool, whether to make grayscale output image
          have and extra degenerate dimension of size 1.
    Returns:
        numpy array, grayscale image.
    '''
    if add_degenerate_dimension:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, None]
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def histogram_equalize_brightness(image):
    '''Improve contrast of RGB or grayscale image via histogram equalization of
    brightness. Preserves degenerate dimension of grayscale image, if any.
    Args:
        image: numpy array, RGB or grayscale image.
    Returns:
        image_out: numpy array, RGB image with improved contrast.
    '''

    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            image_out = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        else:  # image.shape[2] == 1  # gray with degenerate dimension
            image_out = cv2.equalizeHist(image[:, :, 0])[:, :, None]
    else:  # grayscale
        image_out = cv2.equalizeHist(image)

    return image_out


def test_histogram_equalize_brightness(image):
    '''Test histogram brightness equalization on an RGB or grayscale image.
    Args:
        image: numpy array, RGB or grayscale image. Grayscale images are
          allowed to have degenerate dimension.
    Returns:
        None, just plots images.
    '''

    if len(image.shape) == 2 or image.shape[2] == 1:  # grayscale
        fig, axes = plt.subplots(
            1, 2, figsize=(6, 5),
            subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.7, wspace=0.8)
        axf = axes.flat # Iterator over axes.
        ax = next(axf); ax.imshow(image.squeeze(), cmap='gray')
        ax = next(axf); ax.imshow(
            histogram_equalize_brightness(image.squeeze()), cmap='gray')
    else:  # RGB
        image_gray = rgb2gray(image)
        fig, axes = plt.subplots(
            2, 2, figsize=(6, 5),
            subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.7, wspace=0.8)
        axf = axes.flat # Iterator over axes.
        ax = next(axf); ax.imshow(image)
        ax = next(axf); ax.imshow(histogram_equalize_brightness(image))
        ax = next(axf); ax.imshow(image_gray, cmap='gray')
        ax = next(axf); ax.imshow(histogram_equalize_brightness(image_gray), cmap='gray')


def scale_brightness(image, scale_factor=1.0):
    '''Scale brightness of an image, with clamp from above.
    Args:
        image: numpy array, RGB or grayscale image. Grayscale images are
          allowed to have a degenerate dimension, and that dimension is
          preserved when present in input.
        scale_factor: float.
    Returns:
        numpy array, image with brightness scaled.
    '''

    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
        image_out = cv2.cvtColor(
            image, cv2.COLOR_RGB2HSV).astype(dtype=np.float32)
        image_out[:, :, 2] = scale_factor*image_out[:, :, 2]
        image_out[:, :, 2][image_out[:, :, 2] > 255] = 255  # Upper clamp.
        image_out = image_out.astype(dtype=np.uint8)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_HSV2RGB)
    else:  # grayscale
        image_out = scale_factor*image.astype(dtype=np.float32)
        np.clip(image_out, a_min=0.0, a_max=255.0, out=image_out)
        image_out = image_out.astype(dtype=np.uint8)

    return image_out


def normalize(image):
    '''Normalize image, RGB or grayscale.  Puts each pixel into the
    range [-1.0, 1.0].
    Args:
        image: numpy array, image with pixels values in the range [0, 255].
          RGB, grayscale, and grayscale with a degenerate dimension are all
          allowed. Degenerate dimension is preserved when present in input.
    Returns:
        numpy array like input image, but normalized, so
        dtype is float32 instead of a uint8.
    '''
    return ((image - 128.0) / 128.0).astype(np.float32)


def randomly_perturb(
    image,
    brightness_radius=0.3,
    rotation_radius=30.0,
    translation_radius=3,
    shear_radius=3):
    '''Perturb image in brightness, rotation, translation, and shear. The
    *_radius inputs, w.r.t. the infinity norm, are the radii of intervals on
    which perturbation parameters are uniform randomly sampled.
    Inspired in part by
    https://github.com/vxy10/ImageAugmentation
    OpenCV functions
    http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    Args:
        image: numpy array, RGB or grayscale image. Grayscale images are
          allowed to have a degenerate dimension, and that dimension is
          preserved when present in input.
        brightness_radius: float, unitless.
        rotation_radius: float, degrees.
        translation_radius: float, pixels.
        shear_radius: float, pixels.
    Returns:
        image_out: numpy array, perturbed version of the image.
    '''
    row_cnt = image.shape[0]
    column_cnt = image.shape[1]

    # Brightness.
    brightness_factor = np.random.uniform(
        low=1.0 - brightness_radius,
        high=1.0 + brightness_radius)
    image_out = scale_brightness(image, scale_factor=brightness_factor)

    # Rotation.
    rotation_angle = np.random.uniform(
        low=-0.5*rotation_radius, high=0.5*rotation_radius)
    M_rotation = cv2.getRotationMatrix2D(
        (column_cnt/2, row_cnt/2), rotation_angle, 1.0)
        # 旋轉中心, 旋轉角度, 縮放比例


    # Translation.
    dx = np.random.uniform(
      low=-0.5*translation_radius, high=0.5*translation_radius)
    dy = np.random.uniform(
      low=-0.5*translation_radius, high=0.5*translation_radius)
    M_translation = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])

    # Shear.
    points0 = np.float32([[5.0, 5.0], [20.0, 5.0], [5.0, 20.0]])
    point0 = 5.0 + shear_radius*np.random.uniform(low=-0.5, high=0.5)
    point1 = 20 + shear_radius*np.random.uniform(low=-0.5, high=0.5)
    points1 = np.float32([[point0, 5.0], [point1, point0], [5.0, point1]])
    M_shear = cv2.getAffineTransform(points0, points1)

    if len(image_out.shape) == 3 and image_out.shape[2] == 1:
        # Grayscale with degenerate dimension.
        image_out = cv2.warpAffine(
            image_out[:, :, 0], M_rotation, (column_cnt, row_cnt))
        image_out = cv2.warpAffine(
            image_out, M_translation, (column_cnt, row_cnt))
        image_out = cv2.warpAffine(
            image_out, M_shear, (column_cnt, row_cnt))[:, :, None]
    else:  # RGB and grayscale without degenerate dimension.
        image_out = cv2.warpAffine(
            image_out, M_rotation, (column_cnt, row_cnt))
        image_out = cv2.warpAffine(
            image_out, M_translation, (column_cnt, row_cnt))
        image_out = cv2.warpAffine(
            image_out, M_shear, (column_cnt, row_cnt))

        # image_out = rotate(image_out, theta)

    return image_out


def test_randomly_perturb(
    image,
    brightness_radius=0.3,
    rotation_radius=30.0,
    translation_radius=3,
    shear_radius=3):
    '''Test random perturbation on an RGB image.
    Args:
        image: numpy array, RGB or grayscale image. Grayscale images are
            allowed to have degenerate dimension.
        brightness_radius: float.
        rotation_radius: float.
        translation_radius: float.
        shear_radius: float.
    Returns:
        None, just plots images.
    '''
    row_cnt = 6
    column_cnt = 4
    width = 17
    height = 15
    gs = gridspec.GridSpec(row_cnt, column_cnt)
    gs.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
    plt.figure(figsize=(width, height))
    for i in range(row_cnt*column_cnt):
        perturbation = randomly_perturb(
            image,
            brightness_radius,
            rotation_radius,
            translation_radius,
            shear_radius)
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.subplot(row_cnt, column_cnt, i+1)

        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            plt.imshow(perturbation)
        else:  # grayscale
            plt.imshow(perturbation.squeeze(), cmap='gray')

        plt.axis('off')


def preprocess(X):
    '''Prepare numpy array of images for input into CNN.
    
    Args:
        X: numpy array of images, e.g., as X_train,
          X_validate, or X_test.
    
    Returns:
        X_in: numpy array of histogram equalized, normalized, gray images.
    '''

    # TODO shape : 3:->1
    X_in = np.zeros((X.shape[0], X.shape[1], X.shape[2], 3), dtype=np.float32)
    
    for i, image in enumerate(X):
        # image = histogram_equalize_brightness(image)
        
        # TODO model LeNet
        # image = rgb2gray(image)
        # X_in[i, :] = normalize(image)[:, :, None]  # Adds degenerate dimension.

        # model_2 GoogleNet
        # image[:,:,0] = normalize(image[:,:,0])
        # image[:,:,1] = normalize(image[:,:,1])
        # image[:,:,2] = normalize(image[:,:,2])
        X_in[i, :] = image.astype(np.float32) / 128. - 1.

    return X_in



# ----- Dataset Manipulation -----
def shuffle(X, y):
    '''Generate and return randomly shuffled versions of X and y.
    Similar to sklearn.utils.shuffle, but uses numpy.random seed
    for consistency.
    Args:
        X: numpy array, e.g., as X_train, X_validate, or X_test.
        y: numpy array, e.g., as y_train, y_validate, or y_test.
    Returns:œ
        X_shuffled: numpy array of images, e.g., as X_train,
          X_validate, or X_test.
        y_shuffled: numpy array of int labels, e.g., as y_train,
          y_validate, or y_test.
    '''
    ixs_shuffled = np.arange(len(y))
    np.random.shuffle(ixs_shuffled)
    return X[ixs_shuffled, :], y[ixs_shuffled]


def combine(*args):
    '''Combine image classification datasets.
    Args:
        args: a sequence of data pairs, e.g., X0, y0, X1, y1, X2, y2,
          where each X is a numpy array of images and each y is a
          numpy array of class labels.
    Returns:
        X_combined: numpy array of images.
        y_combined: numpy array of class labels.
    '''
    assert len(args) % 2 == 0, 'Must be even number of arguments!'
    input_pair_cnt = int(len(args)/2)
    image_shape = args[0][0, :].shape

    X_dtype = args[0].dtype  # Usu. np.uint8 or np.float32.
    y_dtype = args[1].dtype  # Usu. np.uint8 or str.

    # Count total number of datapoints.
    datapoint_cnt = 0
    for i in range(input_pair_cnt):
        datapoint_cnt += len(args[2*i + 1])

    # Initialize output.
    X_combined = np.zeros((datapoint_cnt, *image_shape), dtype=X_dtype)
    y_combined = np.zeros(datapoint_cnt, dtype=y_dtype)

    # Populate output.
    ix_next = 0
    for i in range(input_pair_cnt):
        ix_next_next = ix_next + len(args[2*i + 1])
        X_combined[ix_next:ix_next_next, :] = args[2*i]
        y_combined[ix_next:ix_next_next] = args[2*i + 1]
        ix_next = ix_next_next

    return X_combined, y_combined


def count_by_label(y):
    '''Count the number of occurrences of each label.
    Args:
        y: numpy array of int labels, e.g., as y_train, y_validate,
          or y_test.
    Returns:
        datapoint_cnt_by_label: dict, where keys are class labels and
          values are the number of respective label occurrences as
          unsigned ints.
    '''
    datapoint_cnt_by_label = {}
    for y_ in y:
        if y_ in datapoint_cnt_by_label:
            datapoint_cnt_by_label[y_] += 1
        else:
            datapoint_cnt_by_label[y_] = 1

    return datapoint_cnt_by_label


def sort_by_class(X, y, class_list, datapoint_cnt_by_label=None):
    '''Sort images by class.
    Args:
        X: numpy array of images, e.g., as X_train, X_validate, or
          X_test.
        y: numpy array of int labels, e.g., as y_train, y_validate,
          or y_test.
        class_list: list of class labels.
        datapoint_cnt_by_label: None or dict, where keys are class
          labels and values are the number of respective label
          occurrences as unsigned ints.  None => computed
          internally, else precompute using
          count_by_label(...).
    Returns:
        images_by_label: dict, where keys are class labels and
          values are numpy arrays of images.
    '''
    image_shape = X[0].shape

    images_by_label = {}  # All images by label.
    if datapoint_cnt_by_label is None:
        datapoint_cnt_by_label = count_by_label(y)

    # Initialize numpy array for each label.
    for label, cnt in datapoint_cnt_by_label.items():
        images_by_label[label] = \
          np.zeros((cnt, *image_shape), dtype=X.dtype)

    # Finish populating output.
    next_image_ixs = { label: 0 for label in class_list }
    for i, label in enumerate(y):
        images_by_label[label][next_image_ixs[label], :] = X[i, :]
        next_image_ixs[label] += 1

    return images_by_label


def balance(
        X, y, class_list, datapoint_cnt_per_class=None, perturb=False):
    '''Create a new dataset with datapoints either pruned or replicated
    such that every class has exactly datapoint_cnt_per_class datapoints.
    Args:
        X: numpy array of images, e.g., as X_train, X_validate, or
          X_test.  Assumed shuffled.
        y: numpy array of int labels, e.g., as y_train, y_validate,
          or y_test.  Assumed shuffled.
        class_list: list of class labels.
        datapoint_cnt_per_class: None or unsigned, optional number
          of data points to use per label. Default is the number of
          datapoints in the most represented class.
        perturb: bool, whether to perturb (modulate brightness, translate,
          rotate, shear) replicated images.
    Returns:
        X_balanced: numpy array of images, where each class has
          exactly datapoint_cnt_per_class datapoints.
        y_balanced: numpy array of unsigned labels, where each
            class has exactly datapoint_cnt_per_class datapoints.
    '''

    class_cnt = len(class_list)
    image_shape = X[0, :].shape
    datapoint_cnt_by_label = count_by_label(y)
    if datapoint_cnt_per_class is None:
        datapoint_cnt_per_class = max(datapoint_cnt_by_label.values())
    images_by_label = sort_by_class(
        X, y, class_list, datapoint_cnt_by_label=datapoint_cnt_by_label)

    # Initialize output.
    datapoint_cnt = datapoint_cnt_per_class*class_cnt
    X_balanced = np.zeros((datapoint_cnt, *image_shape), dtype=np.uint8)
    y_balanced = np.zeros(datapoint_cnt, dtype=np.uint8)

    # Populate output.
    running_datapoint_cnts = { label: 0 for label in class_list }
    for i in range(datapoint_cnt):
        label = class_list[i % class_cnt]
        image = images_by_label[label][i % datapoint_cnt_by_label[label], :]
        if not perturb or \
           running_datapoint_cnts[label] < datapoint_cnt_by_label[label]:
            X_balanced[i, :] = image
        else:
            X_balanced[i, :] = randomly_perturb(image)
        y_balanced[i] = label
        running_datapoint_cnts[label] += 1

    return shuffle(X_balanced, y_balanced)



# ----- Dataset Visualization -----
def truncate_string(s, ubnd, add_ellipses=True):
    '''Return a version of the string that is clamped at the end to
    length `ubnd`.
    Args:
        s: string, string to clamp.
        ubnd: unsigned int, max length of output string.
        add_ellipses: bool, whether to replace the last 3 chars of truncated
          strings with an ellipsis.
    Returns:
      string, clamped version of `s`.
    '''
    s_length = len(s)
    if s_length <= ubnd:
        return s
    else:
        return s[:ubnd-3] + '...'


def class_histograms(
    y_train, y_validate, # y_test,
    class_cnt, suptitle_prefix=''):
    '''Plot histograms of class representation in each dataset.
    Args:
        y_train: numpy array, training labels.
        y_validate: numpy array, validation labels.
        y_test: numpy array, testing labels.
        class_cnt: unsigned, precomputed number of classes.
        suptitle_prefix: string, to describe datasets modified, e.g.,
          by over- or undersampling.
    Returns:
        None
    '''

    bin_cnt = class_cnt

    #fig = plt.figure()
    fig = plt.figure(
        figsize=(13, 10), dpi=80, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.4, wspace=0.7, top=0.9)
    fig.suptitle(r'$\ \ \ $ ' + suptitle_prefix +
        ' Class Counts', fontsize=16, fontweight='bold')

    # Bar chart showing distribution of classes in training set.
    ax = fig.add_subplot(2, 1, 1)  # [rows|columns|plot]
    plt.hist(y_train, bins=bin_cnt, density=False)
    plt.xlabel('', fontsize=14, fontweight='bold')
    #plt.ylabel('Number of Classes', fontsize=14, fontweight='bold')
    # plt.title('Training Set', fontsize=14, fontweight='bold')
    ax.grid(True)

    # Bar chart showing distribution of classes in validation set.
    ax = fig.add_subplot(2, 1, 2)  # [rows|columns|plot]
    plt.hist(y_validate, bins=bin_cnt, density=False)
    # plt.xlabel('', fontsize=14, fontweight='bold')
    #plt.ylabel('Number of Datapoints in Class', fontsize=14, fontweight='bold')
    plt.title('Validation Set', fontsize=14, fontweight='bold')
    ax.grid(True)

    """
    # Bar chart showing distribution of classes in test set.
    ax = fig.add_subplot(3, 1, 3)  # [rows|columns|plot]
    plt.hist(y_test, bins=bin_cnt, normed=False)
    plt.xlabel('Class Label', fontsize=14, fontweight='bold')
    #plt.ylabel('Number of Classes', fontsize=14, fontweight='bold')
    plt.title('Test Set', fontsize=14, fontweight='bold')
    ax.grid(True)
    """


def plot_images(X, y=None, english_labels=None):
    '''Plot all images in a dataset.
    Args:
        X: numpy array of images, e.g., as X_train, X_validate, or
          X_test.
        y: None or indexable of labels, e.g., as y_train,
          y_validate, or y_test.
        english_labels: None or list of strings, the English names
          for labels in y.
    Returns:
        None
    '''
    # ::CAUTION:: RGB images should have type np.uint8. Images cast
    # as a float type will not display correctly with matplotlib.

    image_cnt = X.shape[0]
    column_cnt = 3
    row_cnt = int(np.ceil(image_cnt/column_cnt))
    height = 2.5*row_cnt
    fig, axes = plt.subplots(
        row_cnt, column_cnt, figsize=(15, height),
        subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.7, wspace=0.8)
    axf = axes.flat  # Iterator over axes.
    for i in range(X.shape[0]):
        ax = next(axf)
        if X.shape[3] == 3:
            ax.imshow(X[i, :].squeeze())
        else:
            ax.imshow(X[i, :].squeeze(), cmap='gray')
        if y is not None:
            label = y[i]
            if english_labels is not None:
                full_label = str(label) + ': ' + english_labels[label]
            else:
                full_label = str(label)
            ax.set_title(truncate_string(full_label, ubnd=25))
    for i in range(row_cnt*column_cnt - image_cnt):
        ax = next(axf)
        ax.set_visible(False)  # Hide blank subplot.


def plot_representative_images(
    images_by_label, class_cnt, english_labels=None,
    method='first', image_cnt=None):
    '''Plot one summary image for each class in a dataset.
    Args:
        images_by_label: dict of numpy arrays, e.g., as output by
            sort_by_class(...).
        class_cnt: unsigned, number of image classes.
        english_labels: None or list of strings, the English names
          for labels in y.
        method: string, use as representative
          'first' => first encountered example,
          'mean' => pixelwise mean of first image_cnt images encoutnered,
          'median' => pixelwise median of first image_cnt images encoutnered.
        image_cnt: None or uint, number of images to use for mean or median, if
          applicable. None will use all when 'mean' or 'median are used.
    Returns:
        None
    '''
    # ::CAUTION:: RGB images should have type np.uint8. Images cast
    # as a float type will not display correctly with matplotlib.

    image_shape = next(iter(images_by_label.values()))[0].shape
    if len(image_shape) < 3 or image_shape[2] == 1:
        cmap = 'gray'
    else:
        cmap = 'jet'

    image_cnt = class_cnt
    column_cnt = 3
    row_cnt = int(np.ceil(image_cnt/column_cnt))
    height = 2.5*row_cnt
    fig, axes = plt.subplots(
        row_cnt, column_cnt, figsize=(15, height),
        subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.7, wspace=0.8)
    axf = axes.flat  # Iterator over axes.

    #for label, images in images_by_label.items():
    for label in sorted(images_by_label):
        images = images_by_label[label]
        if image_cnt is None:
            image_cnt = len(images)
        if method == 'mean':
            summary_image = np.mean(
                images[0:image_cnt], axis=0).astype(images.dtype)
        elif method == 'median':
            summary_image = np.median(
                images[0:image_cnt], axis=0).astype(images.dtype)
        else:  # 'first'
            summary_image = images[0, :]
        ax = next(axf)
        ax.imshow(summary_image.squeeze(), cmap=cmap)
        if english_labels:
            full_label = str(label) + ': ' + english_labels[label]
        else:
            full_label = str(label)
        ax.set_title(truncate_string(full_label, ubnd=25))
    for i in range(row_cnt*column_cnt - image_cnt):
        ax = next(axf)
        ax.set_visible(False)  # Hide blank subplot.