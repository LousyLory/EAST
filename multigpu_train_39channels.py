import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import model
import icdar
import cv2
import math
import os
import locality_aware_nms as nms_locality
import lanms
from shapely.geometry import Polygon
from icdar import restore_rectangle
import matplotlib
import matplotlib.pyplot as plt 
import evalEAST
import sys

sys.stdout.flush()

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00005, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'myNNModel/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 14, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 5, '')
tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_float('regC', 1.0, '')

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))
reg_constant = FLAGS.regC

def tower_loss(images, score_maps, geo_maps, training_masks, labels, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)
	f_dat = labels

    model_loss = model.loss(score_maps, f_score,
                            geo_maps, f_geometry,
                            training_masks)
    #total_loss = tf.add_n([model_loss] + 0.7*sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    total_loss = sum([model_loss]) + reg_constant * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        #tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        #tf.summary.image('weight_vis', [v for v in tf.trainable_variables() if 'resnet_v1_50' in v.name][0])
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, f_score, f_geometry, f_dat


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

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

def detect(score_map, geo_map, score_map_thresh=0.1, box_thresh=0.005, nms_thres=0.25):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    print (xy_text[:, ::-1]*4).shape
    print (geo_map[xy_text[:, 0], xy_text[:, 1], :]).shape
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
        if (boxA.is_valid == False):
             return False
        intersection = boxA.intersection(boxB).area
        union = float(boxA.area + boxB.area - intersection)
        return intersection / union > 0.5

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    input_labels = tf.placeholder(tf.float32, shape=[None, None, 4, 2], name='input_labels')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))
    input_labels_split = tf.split(input_labels, len(gpus))
    #x = tf.placeholder(tf.int16, shape=[None, None, 4, 2])
    #y = tf.split(x, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
		il = input_labels_split[i]
                total_loss, model_loss, f_score, f_geometry, f_dat = tower_loss(iis, isms, igms, itms, il, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_fusion')
                grads = opt.compute_gradients(total_loss, var_list=train_var)
                tower_grads.append(grads)
	        #stuff = tf.split(x,len(gpus))[i]

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
    
    variables = slim.get_variables_to_restore()
    #print variables[0].name.split('/')
    #print variables
    var_list = []
    for v in variables:
	if len(v.name.split('/')) == 1:
		var_list.append(v)
	elif v.name.split('/')[1] != "myconv1" or not v.name.find('custom_filter'):
		var_list.append(v)
	else:
		pass
    #var_list=[v for v in variables if v.name.split('/')[1] != "conv1"]
    saver = tf.train.Saver(var_list)
    #print var_list
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    
    '''
    training_list = ["D0006-0285025", "D0017-1592006", "D0041-5370006", "D0041-5370026", "D0042-1070001", "D0042-1070002", "D0042-1070003", "D0042-1070004", "D0042-1070005", "D0042-1070006", "D0042-1070007", "D0042-1070008", "D0042-1070009", "D0042-1070010", "D0042-1070015", "D0042-1070012", "D0042-1070013", "D0079-0019007", "D0089-5235001"]
    validation_list = ["D0090-5242001", "D0117-5755018", "D0117-5755024", "D0117-5755025", "D0117-5755033"]

    with open('Data/cropped_annotations0.txt', 'r') as f:
            annotation_file = f.readlines()
    val_data0 = []
    val_data1 = []
    train_data0 = []
    train_data1 = []
    labels = []
    trainValTest = 2
    for line in annotation_file:
    	if len(line)>1 and line[:11] == 'cropped_img':
                if (len(labels) > 0):
		    if trainValTest == 0:
			train_data1.append(labels)
		    elif trainValTest == 1: 	
			val_data1.append(labels)
                    labels = []
		    trainValTest = 2
        	if line[12:25] in training_list:
		    file_name = "Data/cropped_img_train/"+line[12:].split(".tiff",1)[0]+".tiff"
		    im = cv2.imread(file_name)[:, :, ::-1]
                    train_data0.append(im.astype(np.float32))
		    trainValTest = 0
		elif line[12:25] in validation_list:
	            file_name = "Data/cropped_img_val/"+line[12:].split(".tiff",1)[0]+".tiff"
                    im = cv2.imread(file_name)[:, :, ::-1]
		    val_data0.append(im.astype(np.float32))
		    trainValTest = 1
        elif trainValTest != 2:
	 	annotation_data = line.split(" ")
                if (len(annotation_data) > 2):
		    x, y = float(annotation_data[0]), float(annotation_data[1])
                    w, h = float(annotation_data[2]), float(annotation_data[3])
                    labels.append([[int(x),int(y-h)],[int(x+w),int(y-h)],[int(x+w),int(y)],[int(x),int(y)]])
    if trainValTest == 0:
	train_data1.append(labels)
    elif trainValTest == 1:
	val_data1.append(labels)
    '''  
    init = tf.global_variables_initializer()
    
    if FLAGS.pretrained_model_path is not None:
        print "hereeeee"
	variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #reader = tf.train.NewCheckpointReader("./"+FLAGS.checkpoint_path)
        if FLAGS.restore:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
	    model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Continue training from previous checkpoint here {}'.format(model_path))
            saver.restore(sess, model_path)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
 	sess.run(tf.global_variables_initializer())
	variables_names = [v.name for v in tf.trainable_variables()]
	#print "................."
 	#print variables_names
        #print tf.all_variables()       
        training_list = ["D0006-0285025", "D0017-1592006", "D0041-5370006", "D0041-5370026", "D0042-1070001", "D0042-1070002", "D0042-1070003", "D0042-1070004", "D0042-1070005", "D0042-1070006", "D0042-1070007", "D0042-1070008", "D0042-1070009", "D0042-1070010", "D0042-1070015", "D0042-1070012", "D0042-1070013", "D0079-0019007", "D0089-5235001"]


	a = FLAGS.checkpoint_path[-2]
        data_size = 0
	
        with open('Data/cropped_annotations.txt', 'r') as f:
            annotation_file = f.readlines()
        for line in annotation_file:
            if len(line)>1 and line[:13] == './cropped_img' and line[14:27] in training_list:
                data_size +=1
	print "Char model: " + a
	print "Reg constant: " + str(reg_constant)
	print "Data size: " + str(data_size)
	epoche_size = 3 #ata_size / 32
	print "This many steps per epoche: " + str(epoche_size)
        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers, q_size=10,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus), data_path=a, trainOrVal="train")
        #print "getting the data batches"
	val_data_generator = icdar.get_batch(num_workers=FLAGS.num_readers, q_size=10,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus), data_path=a, trainOrVal="val")
	start = time.time()
        epochsA, ml_list, tl_list = [], [], []
        epochsB, train_fscore, val_fscore = [], [], []
	#print "entering model training"
        for step in range(FLAGS.max_steps):
	    print "this is an iteration............"
            data = next(data_generator)
	    #val_data = next(val_data_generator)
	    
	    if (step % epoche_size == 100):
		#print 'Epochs {:.4f}, ml {:.4f}, tl {:.4f}'.format(float(step)/epoche_size, ml, tl) 
		'''
		train_size = len(train_data0)
                TP, FP, FN = 0.0, 0.0, 0.0
                for i in range(train_size / 128):
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: train_data0[128*i: 128*(i+1)]})
                    labels = sess.run(stuff, feed_dict = {x: train_data1[128*i:128*(i+1)]})
                    TP0, FP0, FN0 = evalEAST.evaluate(score, geometry, labels)
                    TP += TP0
                    FP += FP0
                    FN += FN0
                p_train, r_train = TP / (TP + FP), TP / (TP + FN)
                fscore_train = 2 * p_train * r_train / (p_train + r_train)
		'''
                #for i in range(len(data[0])):
		#    count_right_cache = 0
                #score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: data[0]})
		#p_train, r_train, fscore_train = evalEAST.evaluate(score, geometry, data[5])
		#score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: val_data[0]})
                #p_val, r_val, fscore_val = evalEAST.evaluate(score, geometry, val_data[1])
		'''
                for i in range(len(score)):
		    count_right_cache = 0
		    print score[i].shape, geometry[i].shape
	            boxes = detect(score_map=score[i], geo_map=geometry[i])
                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        for box in boxes:
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            count_wrong += 1
                            num_true_pos = len(data[5][i])
                            for i2 in range(num_true_pos):
                                #print box
                                #print label[i][i2]
                                if (checkIOU(box, label[i][i2]) == True):
                                    count_right_cache += 1
                                    count_wrong -= 1
                    count_posNotDetected += num_true_pos - count_right_cache
                    count_right += count_right_cache
                p_train = (float) (count_right) / (float) (count_right + count_wrong)  # TP / TP + FP
                r_train = (float) (count_right) / (float) (count_right + count_posNotDetected)  # TP / TP + FN
                fscore_train = 2 * (p_train * r_train) / (p_train + r_train)
		print "hi"
	
		score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: val_data[0]})
                for i in range(len(score)):
                    count_right_cache = 0
                    #score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: val_data[0][i]})
                    boxes = detect(score_map=score[i], geo_map=geometry[i])
                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        for box in boxes:
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            count_wrong += 1
                            num_true_pos = len(val_data[1][i])
                            for i2 in range(num_true_pos):
                                #print box
                                #print label[i][i2]
                                if (checkIOU(box, label[i][i2]) == True):
                                    count_right_cache += 1
                                    count_wrong -= 1
                    count_posNotDetected += num_true_pos - count_right_cache
                    count_right += count_right_cache
                p_val = (float) (count_right) / (float) (count_right + count_wrong)  # TP / TP + FP
                r_val = (float) (count_right) / (float) (count_right + count_posNotDetected)  # TP / TP + FN
                fscore_val = 2 * (p_val * r_val) / (p_val + r_val)
                #return precision, recall, fscore
		'''
		#    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: data[0][i]})
		#    fscore_train, p_train, r_train = evalEAST.evaluate(score, geometry, data[5][i])
		#    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: val_data[0]})
                #    fscore_val, p_val, r_val = evalEAST.evaluate(score, geometry, val_data[1])

		print 'Epochs {:.4f}, train fscore {:.4f}, train p {:.4f}, train r {:.4f}, val fscore {:.4f}, val p {:.4f}, val r {:.4f}'.format(float(step)/epoche_size, fscore_train, p_train, r_train, fscore_val, p_val, r_val)            
               
	    #data0 = np.zeros((32,512,512,39)) 
	    ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_score_maps: data[2],
                                                                                input_geo_maps: data[3],
                                                                                input_training_masks: data[4]})
            print ml, tl
	    if step % epoche_size == 0:
		print 'Epochs {:.4f}, ml {:.4f}, tl {:.4f}'.format(float(step)/epoche_size, ml, tl)	
	        #score2, geometry2, dat2 = sess.run([f_score, f_geometry, f_dat], feed_dict={input_images: data[0], input_labels: abc})
                #p_train, r_train, fscore_train = evalEAST.evaluate(score2, geometry2, dat2)
		#print ".."
                #score2, geometry2 = sess.run([f_score, f_geometry], feed_dict={input_images: val_data[0]})
                #p_val, r_val, fscore_val = evalEAST.evaluate(score2, geometry2, val_data[5])
		#print 'Train fscore {:.4f}, train p {:.4f}, train r {:.4f}, val fscore {:.4f}, val p {:.4f}, val r {:.4f}'.format(fscore_train, p_train, r_train, fscore_val, p_val, r_val) 
            
	    if np.isnan(tl):
                print('Loss diverged, stop training')
                break
                       
	    if step % epoche_size == 0: #FLAGS.save_summary_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)
		_, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data0,
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
                summary_writer.add_summary(summary_str, global_step=step)
	#print (time.time() - start) / 3600
if __name__ == '__main__':
    tf.app.run()
