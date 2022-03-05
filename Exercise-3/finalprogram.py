#!/usr/bin/env python

import os
import sys
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'
  MIDDLE_TENSOR_LAYER = 'MobilenetV2/expanded_conv_9/output:0'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)
  
  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    batch_seg_im = self.sess.run(
      self.MIDDLE_TENSOR_LAYER,
      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    segment_im = batch_seg_im[0] 
    return resized_image, seg_map, segment_im


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)



MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'

model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                   download_path)
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)
print('model loaded successfully!')

'''
SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = "lena.png"  #@param {type:"string"}

#_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
              # 'deeplab/g3doc/img/%s.jpg?raw=true')


def run_visualization(url):
  """Inferences DeepLab model and visualizes result."""
  try:
    f = urllib.request.urlopen(url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return
'''
#print('running deeplab on image %s...' % url)
original_im = Image.open('image2.jpg')
resized_im, seg_map, segment_im = MODEL.run(original_im)
vis_segmentation(resized_im, seg_map)

'''
image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
run_visualization(image_url)
'''

############ PCA witn 3 dimensions ############
seg_im = np.array(segment_im)
N = seg_im.shape[0]*seg_im.shape[1]
C = seg_im.shape[-1]
X1 = np.reshape(seg_im, [N, C])
print('First the data have shape: {}'.format(X1.shape))
Xreduced1 = PCA(n_components=3).fit_transform(X1)
print('After PCA data have shape: {}'.format(Xreduced1.shape))
seg_im_reduced = np.reshape(Xreduced1, [seg_im.shape[0], seg_im.shape[1], 3])
print(seg_im_reduced.shape)
plt.imshow(seg_im_reduced)
plt.show()



############ PCA witn 8 dimensions and KMeans ############
seg_im = np.array(segment_im, dtype = np.uint8)
N = seg_im.shape[0]*seg_im.shape[1]
C = seg_im.shape[-1]
X2 = np.reshape(seg_im, [N, C])
print('First the data have shape: {}'.format(X2.shape))
Xreduced2 = PCA(n_components=8).fit_transform(X2)
print('After PCA data have shape: {}'.format(Xreduced2.shape))
seg_im_reduced2 = np.reshape(Xreduced2, [seg_im.shape[0], seg_im.shape[1], 8])


# KMeans Algorithm # 

Seg_im = np.array(seg_im_reduced2)
Im = np.mean(Seg_im, axis = -1)
X = Im.flatten()
K = 2 # Number of clusters
N = len(X)  # Number of pixels
iterations = 30 # Max iteration number of the algorithm
# New centroids 
centroids = np.random.rand(K) * 255
# Print first centroids
for k in range(K):
    print('First Value for Centroid {}: {}'.format(k, centroids[k]))
    for i in range(iterations):
      print('Iteration number - {}'.format(i+1))
      distances_from_centroids = np.zeros([K, N])
      for k in range(K):
        distances_from_centroids[k,:] = (X - centroids[k]*np.ones(N,))**2
      # This vector will store the number of the nearest centroid for every point
      nearest_centroids = np.argmin(distances_from_centroids, axis=0)
      # Compute new centroids
      for k in range(K):
          value_seg = X[nearest_centroids == k]
          if(len(value_seg) == 0):
              continue
          centroids[k] = np.mean(value_seg)
print(nearest_centroids.shape)
segmentation = np.reshape(nearest_centroids, [65, 65])
print(segmentation.shape)
plt.imshow(segmentation)
plt.show()
