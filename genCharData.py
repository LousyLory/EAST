import tensorflow as tf
import numpy as np
import evalEAST
import string
import model
import glob
import cv2
import os

tf.app.flags.DEFINE_string('modelChar', '', '')
#tf.app.flags.DEFINE_string('checkpoint_path', 'east_icdar2015_resnet_v1_50_rbox/', '')
FLAGS = tf.app.flags.FLAGS

dir_path = './detection/'
filepath2 = dir_path + "single_mask/"
filepath3 = dir_path + "draw_1channel/"
#filepath4 = dir_path + "draw_3channel/"
for char in list([FLAGS.modelChar]): #string.ascii_lowercase):
    print FLAGS.model_name
    #filepath = dir_path + char + "2/"
    #if os.path.exists(filepath) == False:
    #    os.mkdir(filepath)
    #if os.path.exists(filepath2) == False:
    #    os.mkdir(filepath2)
    #if os.path.exists(filepath3) == False:
    #    os.mkdir(filepath3)
    #if os.path.exists(filepath4) == False:
    #    os.mkdir(filepath4)
    # load tensorflow model
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state("tmp/east_icdar2015_resnet_v1_50_rbox"+char+"/")
            model_path = os.path.join("tmp/east_icdar2015_resnet_v1_50_rbox"+char+"/", os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)
            for image_name in glob.glob("cropped_img"+"/*"):        
		img = cv2.imread(image_name)
	        if img is None:
	 	    print image_name
		elif os.path.exists(filepath2+image_name[-25:]) == False: # and os.path.exists(filepath3+image_name[-25:]) == True:
		    #generated = True
		    #if os.path.exists(filepath+image_name[-25:]) == False:
		    #	generated = False
                    #    im = np.zeros((img.shape[0], img.shape[1]))
		    #if os.path.exists(filepath2+image_name[-25:]) == False:
		    #    im2 = np.zeros((img.shape[0], img.shape[1]))
	            #else:
		    #    im2 = cv2.imread(filepath2+image_name[-25:])
                    #if os.path.exists(filepath3+image_name[-25:]) == False:
                    #im3 = img.astype(np.uint8).copy() 
                    #else:
                    im3 = cv2.imread(filepath3+image_name[-25:])
                    #if os.path.exists(filepath4+image_name[-25:]) == False:
                    #    im4 = img.astype(np.uint8).copy() 
                    #else:
                    #    im4 = cv2.imread(filepath4+image_name[-25:])
	            # run model and call on eval to return detections
                    img = img[:, :, ::-1]
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [img]})
                    boxes = evalEAST.detect(score_map=score[0], geo_map=geometry[0])
                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
	                for q in boxes:
			    #if generated == False:
				#print q[0]
				#print q[1]
				#print q[2]
				#print q[3]
				#print "omg help....."
				#q[0] = q[3] = (q[0]+q[3])/2
 				#q[1] = q[2] = (q[1]+q[2])/2
				#cv2.polylines(im4, [q.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 255), thickness=2)
                                #cv2.drawContours(im, np.int32([q]), -1, 255, 5)
                            #cv2.fillPoly(im2, np.int32([q]), 255)
			    stuff = q.astype(np.int32).reshape((-1, 1, 2))
			    #print stuff
                            cv2.polylines(im3, [stuff], True, color=(255, 0, 0), thickness=2)
                            #cv2.polylines(im4, [q.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 255), thickness=2)
                    #if generated == False:
		    # 	cv2.imwrite(filepath+image_name[-25:], im)
                    #cv2.imwrite(filepath2+image_name[-25:], im2)
                    cv2.imwrite(filepath3+image_name[-25:], im3)
                    #cv2.imwrite(filepath4+image_name[-25:], im4)
	sess.close()
