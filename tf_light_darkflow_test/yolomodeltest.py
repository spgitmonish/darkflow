#from darkflow.net.build import TFNet
import os
import cv2
import numpy as np
import json
from boxespredict.predict import *

if os.name == 'nt':
	options = {"pbLoad": os.getcwd() + "\\saved_graph\\40375-tiny-yolo-voc-3c.pb", "metaLoad": os.getcwd() + "\\saved_graph\\40375-tiny-yolo-voc-3c.meta", "threshold": 0.1}
else:
	options = {"pbLoad": os.getcwd() + "/saved_graph/40375-tiny-yolo-voc-3c.pb", "metaLoad": os.getcwd() + "/40375-saved_graph/tiny-yolo-voc-3c.meta", "threshold": 0.1}

#tfnet = TFNet(options)
class YOLOTest:
	def __init__(self, options):
		print('\nLoading from .pb and .meta')
		with tf.Session() as sess:
			with tf.gfile.FastGFile(options["pbLoad"], 'rb') as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())
				graph = tf.import_graph_def(graph_def)
				if os.name == "nt":
					self.inp = tf.get_default_graph().get_tensor_by_name('import/input:0')
					self.out = tf.get_default_graph().get_tensor_by_name('import/output:0')
				else:
					self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
					self.out = tf.get_default_graph().get_tensor_by_name('output:0')

				# Feed for testing
				self.feed = dict()

				# Metafile
				with open(options["metaLoad"], 'r') as fp:
					self.meta = json.load(fp)

				# Threshold for prediction filtering
				self.threshold = options["threshold"]

# Create a test object
yolo_test = YOLOTest(options)

for img_file in os.listdir(os.getcwd() + "/sample_img"):
	file_path = os.getcwd() + "/sample_img/" + img_file
	imgcv = cv2.imread(file_path)
	result = return_predict(imgcv, yolo_test)
	print(result)
	print("\n")
