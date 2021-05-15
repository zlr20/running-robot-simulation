import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from yolonets.yolo4 import YoloBody
from yolonets.utils import (DecodeBox, bbox_iou, letterbox_image,
						 non_max_suppression, yolo_correct_boxes)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_yolo(model_path,class_names):
	anchors = [12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401]
	anchors = np.array(anchors).reshape([-1, 3, 2])[::-1,:,:].astype(np.float32)

	yolonet = YoloBody(len(anchors[0]),len(class_names),backbone='mobilenetv1')
	state_dict = torch.load(model_path)
	yolonet.load_state_dict(state_dict)
	
	yolonet = yolonet.to(device)
	yolonet.eval()

	yolodecoders = [DecodeBox(anchors[i], len(class_names),  (416,416)) for i in range(3)]

	return yolonet, yolodecoders

def call_yolo(image,net,decoders,class_names,confidence=0.25,iou=0.3):
	image_shape = image.shape[:2]
	crop_img = cv2.resize(image,(416,416))
	photo = crop_img / 255.0
	photo = np.transpose(photo, (2, 0, 1))
	with torch.no_grad():
		inputs = torch.from_numpy(np.asarray([photo],dtype=np.float32))
		inputs = inputs.to(device)
		outputs = net(inputs)
		output_list = [decoders[i](outputs[i]) for i in range(3)]
		output = torch.cat(output_list, 1)
		batch_detections = non_max_suppression(output, len(class_names),
												conf_thres=confidence,
												nms_thres=iou)
		if batch_detections[0] == None:
			return []
		else:
			batch_detections = batch_detections[0].cpu().numpy()
			top_index = batch_detections[:,4] * batch_detections[:,5] > confidence
			top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
			top_label = np.array(batch_detections[top_index,-1],np.int32)
			top_bboxes = np.array(batch_detections[top_index,:4])
			top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
			top_xmin = top_xmin / 416 * image_shape[1]
			top_ymin = top_ymin / 416 * image_shape[0]
			top_xmax = top_xmax / 416 * image_shape[1]
			top_ymax = top_ymax / 416 * image_shape[0]
			boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
			
			# 返回
			result = []
			for i,c in enumerate(top_label):
				predicted_class = class_names[c]
				score = top_conf[i]
				top, left, bottom, right = boxes[i]
				top = max(0, np.floor(top + 0.5).astype('int32'))
				left = max(0, np.floor(left + 0.5).astype('int32'))
				bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
				right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

			
				result.append({'class':predicted_class,'bbox':[top,left,bottom,right]})
				#result.append([top,left,bottom,right])
			return result

if __name__ == '__main__':
	image=cv2.imread('tmp/1000.png')
	image_copy = image.copy()
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	model_path = './pretrain/6.pth'
	class_names  = ['ball','hole']

	net,decoders = load_yolo(model_path,class_names)
	res = call_yolo(image,net,decoders,class_names,confidence=0.1)

	for info in res:
		pred_class = info['class']
		x1,y1,x2,y2 = info['bbox']
		cv2.rectangle(image_copy,(y1,x1),(y2,x2),(0,255,0),1)
	
	cv2.imwrite('res.png',image_copy)