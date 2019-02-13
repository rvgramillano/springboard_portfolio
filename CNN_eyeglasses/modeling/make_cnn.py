import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report, auc

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    #layer += biases

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def plot_confusion_matrix(cm, classes, mode,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/modeling/plots/%s_cf_aligned.png'%mode, dpi=350)
    plt.clf()

def get_confusion_matrix(labels, predictions, mode):
	final_preds = []
	for pred in predictions:
		final_preds.append(pred['classes'])
	confusion_matrix = tf.confusion_matrix(list(labels), final_preds)
	with tf.Session() as sess:
		cm = sess.run(confusion_matrix)
	# switching order of confusion matrix such that the '1' class (eyeglasses) is on the first row
	cm = np.array([[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]])
	plot_confusion_matrix(np.array(cm), ['Eyeglasses', 'No Eyeglasses'], mode, title='%s Set Confusion Matrix'%mode)
	return final_preds

def get_roc_curve(labels, predictions, mode, color):
	final_pred_probs = []
	for pred in predictions:
		final_pred_probs.append(pred['probabilities'][1])
	fpr, tpr, thresholds = roc_curve(labels, final_pred_probs)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, color=color, label = '%s AUC = %0.2f' % (mode, roc_auc), lw=2.5)
	if mode=='Training':
		plt.title('Receiver Operating Characteristics')
		plt.plot([0, 1], [0, 1],'k--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
	if mode == 'Testing':
		plt.legend(loc= 'lower right')
		plt.tight_layout()
		plt.savefig('/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/modeling/plots/ROC_aligned.png', dpi=350)
		plt.clf()
	return None

def cnn_model(features, labels, mode):

	global dropOut
	global layer1Nodes
	global layer2Nodes
	# Input Layer
	# Celeb images are 28x28 pixels, and have one color channel
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(features["x"], [-1, dim, dim, 1])
	print input_layer.shape
	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 28, 1]
	# Output Tensor Shape: [batch_size, 28, 28, 32]
	conv1, weights_conv1 = new_conv_layer(input=input_layer,num_input_channels=1,filter_size=5,num_filters=layer1Nodes)
	print conv1.shape
	
	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28, 28, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	print pool1.shape
	
	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 14, 14, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 64]
	#conv2 = new_conv_layer(pool1, layer2Nodes, [5,5], tf.nn.relu)
	conv2, weights_conv2 = new_conv_layer(input=pool1,num_input_channels=layer1Nodes,filter_size=5,num_filters=layer2Nodes)
	print conv2.shape
	
	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14, 14, 64]
	# Output Tensor Shape: [batch_size, 7, 7, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	print pool2.shape
	
	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7, 7, 64]
	# Output Tensor Shape: [batch_size, 7 * 7 * 64]
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * layer2Nodes])
	print pool2_flat.shape
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7 * 7 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	print dense.shape
	# Add dropout operation; 0.4 probability that element will be kept
	dropout = tf.layers.dropout(
	    inputs=dense, rate=dropOut, training=mode == tf.estimator.ModeKeys.TRAIN)
	
	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 2]
	logits = tf.layers.dense(inputs=dropout, units=2)
	print logits.shape
	predictions = {
	    # Generate predictions (for PREDICT and EVAL mode)
	    "classes": tf.argmax(input=logits, axis=1),
	    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	    # `logging_hook`.
	    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
	    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	
	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
	loss = tf.losses.softmax_cross_entropy(
	    onehot_labels=onehot_labels, logits=logits)
	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
	  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	  train_op = optimizer.minimize(
	      loss=loss,
	      global_step=tf.train.get_global_step())
	  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	    "accuracy": tf.metrics.accuracy(
	        labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.INFO)

dim = 28
train_num = 61531 #80% of 76914
test_num = 15383

data = np.load("/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/CelebA/CelebA_70K_align.npz")
labels = data['labels']
celebData = data['imageData']
imageNames = data['imageNames']

trains_images = celebData[0:train_num,:,:]#getting the training sets
train_images_labels = labels[0:train_num,:]

test_images = celebData[train_num:train_num+test_num,:,:]#getting the training sets
test_images_labels = labels[train_num:train_num+test_num,:]

#flattening the input array and reshaping the labels as per requirement of the tnesorflow algo
trains_images = trains_images.reshape([train_num,dim**2])
test_images = test_images.reshape([test_num,dim**2])
train_images_labels = train_images_labels.reshape([train_num,])
test_images_labels = test_images_labels.reshape([test_num,])

#standardizing the image data set with zero mean and unit standard deviation
trains_images = preprocessing.scale(trains_images)
test_images = preprocessing.scale(test_images)

# set hyperparameters
global dropOut
global layer1Nodes
global layer2Nodes
dropOut = 0.4
layer1Nodes = 32
layer2Nodes = 64
#saving the trained model on this path
modelName = "/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/modeling/celeb_convnet_model_aligned"+str(dropOut)+str(layer1Nodes)+str(layer2Nodes)
# Load training and eval data
train_data = np.asarray(trains_images, dtype=np.float32)  # Returns np.array
train_labels = train_images_labels
test_data = np.asarray(test_images, dtype=np.float32)  # Returns np.array 
test_labels = test_images_labels

# Create the Estimator
celeb_classifier = tf.estimator.Estimator(
  model_fn=cnn_model, model_dir=modelName)

# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=1000)

# Train the model
train_input_fn= tf.estimator.inputs.numpy_input_fn(
  x={"x": train_data},
  y=train_labels,
  batch_size=100,
  num_epochs=None,
  shuffle=True)
celeb_classifier.train(
  input_fn=train_input_fn,
  steps=2000)#epoch count

# Evaluate the training set and print results
Train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": train_data},
  y=train_labels,
  num_epochs=1,
  shuffle=False)

train_results = celeb_classifier.evaluate(input_fn=Train_input_fn)
print("Training set accuracy", train_results)
train_predictions = list(celeb_classifier.predict(input_fn=Train_input_fn))
train_labels = list(train_labels)
train_preds = get_confusion_matrix(train_labels, train_predictions, 'Training') 
cr_train = classification_report(train_labels, train_preds)
print '-----------------------TRAINING SET-----------------------'
print cr_train

# Evaluate the Test set and print results
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": test_data},
  y=test_labels,
  num_epochs=1,
  shuffle=False)
test_results = celeb_classifier.evaluate(input_fn=test_input_fn)
print("Test accuracy" ,test_results)
test_predictions = list(celeb_classifier.predict(input_fn=test_input_fn))
test_labels = list(test_labels)
test_preds = get_confusion_matrix(test_labels, test_predictions, 'Testing')
cr_test = classification_report(test_labels, test_preds)
print '-----------------------TESTING SET-----------------------'
print cr_test

# get ROC curves for both training and testing set
get_roc_curve(train_labels, train_predictions, 'Training', 'royalblue')
get_roc_curve(test_labels, test_predictions, 'Testing', 'firebrick')




