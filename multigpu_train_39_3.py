import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import model_39channels as model
import icdar
import cv2
import math
import os
import locality_aware_nms as nms_locality
import lanms
from shapely.geometry import Polygon
from icdar import restore_rectangle
#import evalEAST
import sys
import random
import string

sys.stdout.flush()

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00005, '')
tf.app.flags.DEFINE_integer('max_steps', 2000, '')
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

training_list = ["D0006-0285025", "D0017-1592006", "D0041-5370006", "D0041-5370026", "D0042-1070001", "D0042-1070002", \
                 "D0042-1070003", "D0042-1070004", "D0042-1070005", "D0042-1070006", "D0042-1070007", "D0042-1070009", \
                 "D0042-1070010", "D0042-1070015", "D0042-1070012", "D0042-1070013", "D0079-0019007", "D0089-5235001"]

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

def i_am_testing(images):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        f_score, f_geometry = model.model(images, is_training=True)
    #return f_score, f_geometry

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
    xy_text = np.argwhere(score_map > score_map_thresh)
    if len(xy_text) < 1:
	return None
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

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 39], name='input_images')
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

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))
    input_labels_split = tf.split(input_labels, len(gpus))
    
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
                total_loss, model_loss, f_score, f_geometry, _ = tower_loss(iis, isms, igms, itms, il, reuse_variables)
                #f_score, f_geometry = i_am_testing(iis)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                #print "below..."
                #batch_norm_updates_op = tf.group(*[op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope) if 'resnet_v1_50/block4' in op.name or 'resnet_v1_50/block3' in op.name or 'feature_fusion' in op.name])
                #print "above..."
                reuse_variables = True
                #print "below.."
                train_var = [var for var in tf.trainable_variables() if 'resnet_v1_50/block1' in var.name]
                #train_var = [var for var in tf.trainable_variables() if 'resnet_v1_50/block4' in var.name]
                #train_var += [var for var in tf.trainable_variables() if 'feature_fusion/Conv_7' in var.name]
                #train_var += [var for var in tf.trainable_variables() if 'feature_fusion/Conv_8' in var.name]
                #train_var += [var for var in tf.trainable_variables() if 'feature_fusion/Conv_9' in var.name]
                #print train_var
                #print "above..."
                train_var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_fusion')
                grads = opt.compute_gradients(total_loss, var_list=train_var)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    #train_var = [var for var in tf.trainable_variables() if ('resnet_v1_50/block3' in var.name or 'resnet_v1_50/block4' in var.name or 'feature_fusion' in var.name)]
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
    
    #####################################################################################################################
    # BLOCK MODIFIED BY ME
    #variables = slim.get_variables_to_restore()
    #var_list = []
    #for v in variables:
    #    if len(v.name.split('/')) == 1:
    #            var_list.append(v)
    #    elif v.name.split('/')[1] != "myconv1" or not v.name.find('custom_filter'):
    #            var_list.append(v)
    #    else:
    #            pass
    #saver = tf.train.Saver(var_list)
    saver = tf.train.Saver(tf.global_variables())
    saver_restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # removing the first conv layer
    #del saver_restore_vars[1]
    #saver_to_restore = tf.train.Saver(saver_restore_vars)
    #####################################################################################################################
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    
    init = tf.global_variables_initializer()
    #print '>> trainable variables: ',slim.get_trainable_variables()
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    #my_char_l = "5"
    #my_char_U = ""
    data_size = 0
    train_data_indices = []
    list_of_img_pos = []
    with open('./cropped_annotations_3.txt', 'r') as f:
        annotation_file = f.readlines()
    #with open('Data/cropped_annotations_new/cropped_annotations' + my_char_U + '.txt', 'r') as f:
    #    annotation_file += f.readlines()
    idx = 0
    for line in annotation_file:
	if len(line)>1 and line[:13] == './cropped_img':# and str(line[14:27]) in training_list:
            data_size +=1
            train_data_indices.append(idx)
            list_of_img_pos.append(line[14:].split(".")[0]+".tiff")
        idx += 1
    list_of_img_all = os.listdir('./cropped_img')
    list_of_img_neg = np.array(list(set(list_of_img_all) - set(list_of_img_pos)))
    #print "Char model: " + my_char_U + my_char_l
    #print "Data size: " + str(data_size)
    epoch_size = data_size / (16 * 2)
    #print epoch_size
    print "This many steps per epoch: " + str(epoch_size)

    #list_of_img_neg_char = os.listdir('Data/j')

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
	    print '>> Checkpoint path: ', FLAGS.checkpoint_path
	    print '>> second stuff: ', os.path.basename(ckpt_state.model_checkpoint_path)
	    #all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1]
	    var1 = saver_restore_vars[1]
	    del saver_restore_vars[1]
	    var2 = saver_restore_vars[422]
	    del saver_restore_vars[422]
	    #names = [var.name for var in saver_restore_vars]
	    saver_to_restore = tf.train.Saver(saver_restore_vars)	
	    #print '>> global vars: ', names.index('resnet_v1_50/conv1/weights/ExponentialMovingAverage:0')#[var.name for var in tf.global_variables()]
	    model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
	    # originally saver.restore(sess, model_path)
            saver_to_restore.restore(sess, model_path)
	    init_new_vars_op = tf.initialize_variables([var1, var2])
	    sess.run(init_new_vars_op)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)
        #print "below:"
        #tvars = tf.trainable_variables()
        #g_vars = [var for var in tvars if 'resnet_v1_50/block4' in var.name]
        #print g_vars
        #print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_50')
        #return
        print FLAGS.learning_rate
        print reg_constant
        for step in range(24*epoch_size):
            ### Generate Dwata ###
            data = [], [], [], [], []
            np.random.shuffle(train_data_indices)
            num_im = 0
            actual_num_im = 0
	    list_of_chars = list(string.ascii_lowercase)+[str(x) for x in range(10)]
            while len(data[0]) < 32:
                prob = 1#np.random.random(1)[0]
                if prob > 0.49:
                    i = train_data_indices[num_im]
                    im_fn = "./cropped_img/"+annotation_file[i][14:].split(".tiff",1)[0]+".tiff"
		    #print im_fn
                    im = cv2.imread(im_fn)
		    ################################################################################
                    # adding rest of the channels
                    for ids_c in range(len(list_of_chars)):
                        crop_dir = '/mnt/nfs/work1/elm/ray/evaluation/EAST_cropped/'+list_of_chars[ids_c]+'/'
                        filename = crop_dir+annotation_file[i][14:].split(".tiff",1)[0]+".tiff"
                        pad = cv2.imread(filename)
                        pad = pad[:,:,0]
                        pad = np.expand_dims(pad, axis=2)
                        im = np.append(im, pad, axis = 2)
                    ################################################################################
		    ################################################################################
                    if im is not None:
                        r, c, _ = im.shape
		        text_polys = []
                        text_tags = []
                        if int(annotation_file[i+1]) > 0:
                            for idx in range(i+2,i+2+int(annotation_file[i+1])):
                                annotation_data = annotation_file[idx]
                                annotation_data = annotation_data.split(" ")
                                x, y = float(annotation_data[0]), float(annotation_data[1])
		                w, h = float(annotation_data[2]), float(annotation_data[3])
		                text_polys.append([list([int(x),int(y-h)]),list([int(x+w),int(y-h)]),list([int(x+w),int(y)]),list([int(x),int(y)])])
                                text_tags.append(False)
                        score_map, geo_map, training_mask = icdar.generate_rbox((int(r), int(c)), np.array(text_polys), np.array(text_tags))
                        data[0].append(im[:, :, ::-1].astype(np.float32))
                        data[1].append(im_fn)
                        data[2].append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                        data[3].append(geo_map[::4, ::4, :].astype(np.float32))
                        data[4].append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
                        actual_num_im += 1  
                    num_im += 1
           
                else:
                    im_fn = np.random.choice(list_of_img_neg)
		    ################################################################################
                    # adding rest of the channels
                    #for i in range(len(list_of_chars)):
                    crop_dir = '/mnt/nfs/work1/elm/ray/evaluation/EAST_single_cropped/'
                    filename = crop_dir+annotation_file[i][14:].split(".tiff",1)[0]+".tiff"
                    pad = cv2.imread(filename)
                    pad = pad[:,:,0]
                    pad = np.expand_dims(pad, axis=2)
                    im = np.append(im, pad, axis = 2)
                    ################################################################################
                    #    im_fn = np.random.choice(list_of_img_neg_char)
                    #    im_mini = cv2.imread("Data/j/" + im_fn)
		    # 	r0, c0, _ = im_mini.shape
                    #     im = np.zeros((512, 512, 3), dtype=np.uint8)
 		    #	ra, rb, ca, cb = 256-r0/2, 256+(r0+1)/2, 256-c0/2, 256+(c0+1)/2
                    #    im[ra:rb, ca:cb, :] = im_mini.copy()
                    if im is not None:
                        r, c, _ = im.shape
                        score_map, geo_map, training_mask = icdar.generate_rbox((int(r), int(c)), np.array([]), np.array([]))
                        data[0].append(im[:, :, ::-1].astype(np.float32))
                        data[1].append(im_fn)
                        data[2].append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                        data[3].append(geo_map[::4, ::4, :].astype(np.float32))
                        data[4].append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
       
            ### Run model ###
    	    ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                    input_score_maps: data[2],
                                                                                    input_geo_maps: data[3],
                                                                                    input_training_masks: data[4]})
            epoch = step / epoch_size
            batch_num = step % epoch_size   
            if step % (epoch_size/3) == 0:   
                print "Epoch no.: " + str(epoch) + " batch no.: " + str(batch_num) + " loss: " + str(ml)
                print "Epoch no.: " + str(epoch) + " batch no.: " + str(batch_num) + " loss: " + str(tl)
    	    if step % (epoch_size/2) == 0:
		#print "Epoche: " + str(step / (epoch_size/2))
		saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)
    	        _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                                 input_score_maps: data[2],
                                                                                                 input_geo_maps: data[3],
                                                                                                 input_training_masks: data[4]})
                summary_writer.add_summary(summary_str, global_step=step)
            if False:
                count_right = 0
                count_wrong = 0
                count_posNotDetected = 0
                im0 = cv2.imread("Data/maps/D0117-5755036.tiff")[:, :, ::-1]
                w, h, _ = im0.shape
                slide_window = 300
                crop_size = 512
                crop_center = (256, 256)
                num_rows, num_cols = int(np.ceil(w/slide_window)), int(np.ceil(h/slide_window))
                print num_cols
		for rot in [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]:
                    im = cv2.imread("Data/maps/D0117-5755036.tiff")[:, :, ::-1]
                    boxes_one_rot = []  
		    count = 0
                    while count < num_rows * num_cols:
                        images, data2, data3, data4 = [], [], [], []
                        for k in range(16):
                            i = (count + k) / num_rows
                            j = (count + k) % num_cols
                    
                            temp = im[slide_window*i:slide_window*i+crop_size, \
                                      slide_window*j:slide_window*j+crop_size, ::-1]
                            w2, h2, _ = temp.shape
                            if w2 < crop_size or h2 < crop_size:
                                result = np.zeros((crop_size,crop_size,3))
                                result[:w2,:h2] = temp
                                temp = result
                            M = cv2.getRotationMatrix2D(crop_center,rot,1.0)
                            temp = cv2.warpAffine(temp, M, (crop_size, crop_size))
                            images.append(temp)
			    score_map, geo_map, training_mask = icdar.generate_rbox((int(crop_size), int(crop_size)), np.array([]), np.array([]))
                            data2.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                            data3.append(geo_map[::4, ::4, :].astype(np.float32))
                            data4.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
                        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: images, input_score_maps:data2,
                                                                                                 input_geo_maps: data3,
                                                                                                 input_training_masks: data4})
                        for k in range(16):
                            i = (count + k) / num_rows
                            j = (count + k) % num_cols
                            boxes = detect(score_map=score[j], geo_map=geometry[j], score_map_thresh=0.01, box_thresh=0.01, nms_thres=0.01)
                            if boxes is not None:
                                boxes = boxes[:, :8].reshape((-1, 4, 2))
                                for box in boxes:
                                    M_inv = cv2.getRotationMatrix2D(crop_center,-1*rot,1)
                                    box[0] = M_inv.dot(np.array((box[0,0], box[0,1]) + (1,)))
                                    box[1] = M_inv.dot(np.array((box[1,0], box[1,1]) + (1,)))
                                    box[2] = M_inv.dot(np.array((box[2,0], box[2,1]) + (1,)))
                                    box[3] = M_inv.dot(np.array((box[3,0], box[3,1]) + (1,)))
                                    box = sort_poly(box.astype(np.int32))
                                    box[0,0] = box[0,0] + j * slide_window
                                    box[0,1] = box[0,1] + i * slide_window
                                    box[1,0] = box[1,0] + j * slide_window
                                    box[1,1] = box[1,1] + i * slide_window
                                    box[2,0] = box[2,0] + j * slide_window
                                    box[2,1] = box[2,1] + i * slide_window
                                    box[3,0] = box[3,0] + j * slide_window
                                    box[3,1] = box[3,1] + i * slide_window
                    boxes_one_rot.append(box)
                    boxes_single_rot = np.zeros((len(boxes_one_rot), 9))
                    boxes_single_rot[:, :8] = np.array(boxes_one_rot).reshape((-1, 8))
                    boxes_single_rot[:, 8] = 1
                    labels += boxes_single_rot.tolist()                                               
                boxes = lanms.merge_quadrangle_n9(np.array(labels), nms_thres)
                annotation = np.load("/mnt/nfs/work1/elm/ray/new_char_anots_ncs/" + "j" + "/" + "D0117-5755036" + ".npy").item()
                ### Compute the TP, FP, FN info for each image
                count_right_cache = 0
                boxes = boxes[:, :8].reshape((-1, 4, 2))
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
	    

if __name__ == '__main__':
    tf.app.run()
