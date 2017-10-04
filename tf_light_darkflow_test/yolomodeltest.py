from darkflow.net.build import TFNet
import os
import cv2
import numpy as np

options = {"pbLoad": os.getcwd() + "/built_graph/tiny-yolo-voc-3c.pb", "metaLoad": os.getcwd() + "/built_graph/tiny-yolo-voc-3c.meta", "threshold": 0.1}

tfnet = TFNet(options)

for img_file in os.listdir(os.getcwd() + "/sample_img"):
	file_path = os.getcwd() + "/sample_img/" + img_file
	imgcv = cv2.imread(file_path)
	result = tfnet.return_predict(imgcv)
	print(result)
	print("\n")



