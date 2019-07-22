import tensorflow as tf
from keras import backend as K
#CUSTOM METRICS

def dice_coefficient(y_true, y_pred, smooth=1e-6):

    """ Loss function base on dice coefficient.
    ----------
    y_true : keras tensor containing target mask.
    y_pred : keras tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing dice loss.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def labels_to_one_hot(y_true, num_classes=1):
    
    """
    https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/loss_segmentation.html#dice_plus_xent_loss
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(y_true)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(y_true, output_shape)

    # squeeze the spatial shape
    y_true = tf.reshape(y_true, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(y_true)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    y_true = tf.to_int64(y_true)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, y_true], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(y_true, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

    
def generalised_dice(y_true, y_pred,
                          type_weight='Square'):
                              
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    y_pred = tf.cast(y_pred, tf.float32)
    if len(y_true.shape) == len(y_pred.shape):
        y_true = y_true[..., -1]
    one_hot = labels_to_one_hot(y_true, tf.shape(y_pred)[-1])

    ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    intersect = tf.sparse_reduce_sum(one_hot * y_pred,
                                         reduction_axes=[0])
    seg_vol = tf.reduce_sum(y_pred, 0)
    
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
#    generalised_dice_denominator = \
#         tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return generalised_dice_score


def generalised_dice_loss(y_true, y_pred):
    return 1 - generalised_dice(y_true, y_pred)
    
    
    
def tversky(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    
    """ Tversky loss function.
    Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour
    Parameters
    ----------
    y_true : keras tensor 
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    tversky = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    
    return tversky
    
def tversky_loss(y_true, y_pred):
    return -tversky(y_true, y_pred)
    
    
    
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    """
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
    

