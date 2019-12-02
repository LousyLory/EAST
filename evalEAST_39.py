import os
import copy
import cv2
import numpy as np
import lanms
import model_39channels as model
import logging
import tensorflow as tf
from shapely.geometry import Polygon
from shapely.affinity import rotate
from icdar import restore_rectangle
import datetime
import string
import sys
tf.app.flags.DEFINE_string('visualize', 'False', '')
tf.app.flags.DEFINE_string('model_name', '', '')
#tf.app.flags.DEFINE_string('checkpoint_path', 'east_icdar2015_resnet_v1_50_rbox/', '')
FLAGS = tf.app.flags.FLAGS

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def detect(score_map, geo_map, score_map_thresh, box_thresh, nms_thres):
    '''

    '''
    if len(score_map.shape) == 3:
        score_map = score_map[:, :, 0]
        #geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, color=np.array((255,0,0)))
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes	

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def checkIOU(boxA, boxB):
        boxA = Polygon([(boxA[0,0], boxA[0,1]), (boxA[1,0], boxA[1,1]), (boxA[2,0], boxA[2,1]), (boxA[3,0], boxA[3,1])])
        boxB = Polygon([(boxB[0][0], boxB[0][1]), (boxB[1][0], boxB[1][1]), (boxB[2][0], boxB[2][1]), (boxB[3][0], boxB[3][1])])
        if (boxA.is_valid == False or boxB.is_valid == False):
             return False
        intersection = boxA.intersection(boxB).area
        union = float(boxA.area + boxB.area - intersection)
        return intersection / union > 0.5

def evaluate(score, geometry, label):
        """ This function is more for evaluating model during training """
        count_right = 0
        count_wrong = 0
        count_posNotDetected = 0
        print len(score)
        print len(label)
        for i in range(len(label)):
            count_right_cache = 0
	    boxes = detect(score_map=score[i], geo_map=geometry[i])
            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                for box in boxes:
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    count_wrong += 1
                    num_true_pos = len(label[i])
                    for i2 in range(num_true_pos):
			#print box
			print label[i][i2]
                        if (checkIOU(box, label[i][i2]) == True):
                            count_right_cache += 1
                            count_wrong -= 1
                            count_posNotDetected += num_true_pos - count_right_cache
                            count_right += count_right_cache
        precision = (float) (count_right) / (float) (count_right + count_wrong)  # TP / TP + FP
        recall = (float) (count_right) / (float) (count_right + count_posNotDetected)  # TP / TP + FN
        fscore = 2 * (precision * recall) / (precision + recall)
	return precision, recall, fscore

def evaluateModel(model_ver, model_path, angles, imname_list, im_dir, annot_dir, results_dir, nms_thres, score_map_thresh, box_thresh, inspect_mode=False, calc_p_r=False):
    mask_dir = '/mnt/nfs/work1/elm/ray/evaluation/EAST_mask/'
    list_of_dirs = list(string.ascii_lowercase)+[str(x) for x in range(10)]
    print "Evaluating EAST model, separately from training process..."
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 39], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=True)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(model_path)
            model_path = os.path.join(model_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)
            count_right = 0
            count_wrong = 0
            count_posNotDetected = 0
	    for imName in imname_list:
		print imName
		labels = []
		im00 = cv2.imread(im_dir + imName + ".tiff")[:, :, ::-1]
                w, h, _ = im00.shape
                im0 = im00[:, :, :]
                im0 = cv2.resize(im0, (h, w))
                slide_window = 400
                crop_size = 512
                crop_center = (256, 256)
                num_rows, num_cols = int(np.ceil(w/slide_window)), int(np.ceil(h/slide_window))
		for rot in angles:
	            im = cv2.imread(im_dir + imName + ".tiff")#[:, :, ::-1]
		    for character_indices in range(len(list_of_dirs)):
		    	filename = mask_dir+list_of_dirs[character_indices]+'/'+imName+".tiff"
		    	mask_im = cv2.imread(filename)[:,:,0]
		    	mask_im = np.expand_dims(mask_im, axis=2)
		    	im = np.append(im, mask_im, axis=2)
                    boxes_one_rot = [] 	
		    for i in range(num_rows):
			images = []
			for j in range(num_cols):
			    temp = im[slide_window*i:slide_window*i+crop_size, \
				      slide_window*j:slide_window*j+crop_size, :]
			    w2, h2, _ = temp.shape
			    if w2 < crop_size or h2 < crop_size:
			 	result = np.zeros((crop_size,crop_size,39))
				result[:w2,:h2] = temp
				temp = result
                            M = cv2.getRotationMatrix2D(crop_center,rot,1.0)
                            temp = cv2.warpAffine(temp, M, (crop_size, crop_size))
			    images.append(temp)
			score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: images})
                        for j in range(num_cols):
			    boxes = detect(score_map=score[j], geo_map=geometry[j], score_map_thresh=score_map_thresh, box_thresh=box_thresh, nms_thres=nms_thres)
			    if boxes is not None:
                                boxes = boxes[:, :8].reshape((-1, 4, 2))
                                for box in boxes:
                                    M_inv = cv2.getRotationMatrix2D(crop_center,-1*rot,1)
                                    box[0] = M_inv.dot(np.array((box[0,0], box[0,1]) + (1,)))
                                    box[1] = M_inv.dot(np.array((box[1,0], box[1,1]) + (1,)))
                                    box[2] = M_inv.dot(np.array((box[2,0], box[2,1]) + (1,)))
                                    box[3] = M_inv.dot(np.array((box[3,0], box[3,1]) + (1,)))
				    box[0,0] = box[0,0] + j * slide_window
                                    box[0,1] = box[0,1] + i * slide_window
                                    box[1,0] = box[1,0] + j * slide_window
                                    box[1,1] = box[1,1] + i * slide_window
                                    box[2,0] = box[2,0] + j * slide_window
                                    box[2,1] = box[2,1] + i * slide_window
                                    box[3,0] = box[3,0] + j * slide_window
                                    box[3,1] = box[3,1] + i * slide_window
				    # setting up the actual orientation expected
				    box_f = copy.copy(box)
				    box_f[0,0] = box[3,0]
				    box_f[0,1] = box[3,1]
                                    box_f[1,0] = box[2,0]
                                    box_f[1,1] = box[2,1]
                                    box_f[2,0] = box[1,0]
                                    box_f[2,1] = box[1,1]
                                    box_f[3,0] = box[0,0]
                                    box_f[3,1] = box[0,1]
				    boxes_one_rot.append(box_f)
                    boxes_single_rot = np.zeros((len(boxes_one_rot), 9))
                    boxes_single_rot[:, :8] = np.array(boxes_one_rot).reshape((-1, 8))
                    boxes_single_rot[:, 8] = 1
                    labels += boxes_single_rot.tolist()
                    if inspect_mode == True:
                        boxes_single_rot = lanms.merge_quadrangle_n9(boxes_single_rot, lanms_thresh).astype('int16')
                        for box in boxes_single_rot:
			    pts = np.array(box[:8], dtype=np.int32).reshape((4,2))
                            cv2.polylines(im0[-w/2, -h/2, ::-1], [pts], True, color=(255, 255, 0), thickness=4)
                        cv2.imwrite(results_dir+imName+model_ver+str(int(rot))+".tiff", im[-w/2, -w/2, ::-1])
                        np.savetxt(results_dir+imName+model_ver+str(int(rot))+".txt", boxes_single_rot, '%5.1f', delimiter=",")
                ### Plot out and save the text detections for each image                                                
                boxes = lanms.merge_quadrangle_n9(np.array(labels), nms_thres)
		if not os.path.isdir('Data/final_detections_npy/'):
			os.mkdir('Data/final_detections_npy/')
		if not os.path.isdir('Data/final_detections_npy/'+model_ver):
                        os.mkdir('Data/final_detections_npy/'+model_ver)
		np.save('Data/final_detections_npy/'+model_ver+'/'+imName+'.npy', boxes)
                #for box in boxes:
		#    pts = np.array(box[:8], dtype=np.int32).reshape((4,2))
                #    cv2.polylines(im00[:, :, ::-1], [pts], True, color=(255, 0, 0), thickness=4)
                #annotation = np.load("/mnt/nfs/work1/elm/ray/new_char_anots_ncs/" + "j" + "/" + imName + ".npy").item()
                #for index in range(len(annotation)):
                #    vertices = annotation[index]["vertices"]
                #    vertices = [np.array(vertices, dtype=np.int32)]
                #    cv2.polylines(im00[:, :, ::-1], vertices, True, color=(0, 0, 255), thickness=2)
                #cv2.imwrite(results_dir+imName+model_ver+".tiff", im00[:, :, ::-1])
                #np.savetxt(results_dir+imName+model_ver+".txt", boxes, delimiter=",")
                ### Compute the TP, FP, FN info for each image
                if calc_p_r:
                    count_right_cache = 0
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    annotation = np.load(annot_dir + imName + ".npy").item()
                    num_true_pos = len(annotation)
                    for box in boxes:
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                            continue
                        k = 0
                        idx = 0
                        count_wrong += 1
                        while (idx < num_true_pos):
                            if k in annotation: 
                                proposed_label = annotation[k]['vertices']
                                if len(proposed_label) == 4:
                                    x3, y3, x2, y2, x1, y1, x0, y0 = proposed_label[0][0], proposed_label[0][1], proposed_label[1][0], proposed_label[1][1], \
                                                                     proposed_label[2][0], proposed_label[2][1], proposed_label[3][0], proposed_label[3][1]
                                    if (checkIOU(box, [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]) == True):
                                        count_right_cache += 1
                                        count_wrong -= 1
                                        break 
                                idx += 1
                            k += 1
                    count_posNotDetected += num_true_pos - count_right_cache
                    count_right += count_right_cache
            	    precision = (float) (count_right) / (float) (count_right + count_wrong)  # TP / TP + FP
            	    recall = (float) (count_right) / (float) (count_right + count_posNotDetected)  # TP / TP + FN
            	    fscore = 2 * (precision * recall) / (precision + recall)
            	    print "Precision, recall, fscore: " + str(precision) + ", " + str(recall) + ", " + str(fscore)

def main(argv=None):
        # Set up parameters here
        training_list = ["D0041-5370026","D0006-0285025", "D0042-1070003", "D0041-5370026", "D0017-1592006", \
			"D0041-5370006", "D0042-1070001", "D0042-1070002", "D0042-1070004", "D0042-1070005", \
			"D0042-1070006", "D0042-1070007", "D0042-1070009", "D0042-1070010", "D0042-1070015", \
			"D0042-1070012", "D0042-1070013", "D0079-0019007", "D0089-5235001"]
	validation_list = ["D0090-5242001", "D0117-5755018", "D0117-5755024", "D0117-5755033", "D0117-5755025"]
	test_list = ["D0117-5755035"]#, "D0117-5755036", "D5005-5028052", "D5005-5028054", \
		     #"D5005-5028097", "D5005-5028100", "D5005-5028102", "D5005-5028149"]
        angles = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0] #-90.0, -75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0] #-95.0, -85.0, -75.0, -65.0, -55.0, -45.0, -35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
        model_ver = "_39_channel_1"
        model_path = "tmp/east_icdar2015_resnet_v1" + model_ver
	imname_list = [sys.argv[1]]#["D0117-5755035"]#, "D0089-5235001"]  #training_list + validation_list + test_list 
        im_dir = "Data/maps/"
        annot_dir = "Data/maps/"
        results_dir = "Results/"
	lanms_thresh = 0.1
        score_map_thresh = 0.1
        box_thresh = 0.025
        inspect_mode = False
        calc_p_r = True
        print "Model used: " + model_path
        print "lanms threshold used: " + str(lanms_thresh)
        print "score map threshold used: " + str(score_map_thresh)
        print "box threshold: " + str(box_thresh)
        start_time = datetime.datetime.now()
        evaluateModel(model_ver, model_path, angles, imname_list, im_dir, annot_dir, results_dir, lanms_thresh, score_map_thresh, box_thresh, inspect_mode, calc_p_r)
        print "Code execution time in minutes: " + str((datetime.datetime.now()-start_time).seconds / 60)

if __name__ == '__main__':
    tf.app.run()

