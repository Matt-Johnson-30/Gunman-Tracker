import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', './outputs/tracker.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.35, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    # load saved tensorflow model
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-final-guns-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(416)
        height = int(416)
        fps = int(10)
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
	
    frame_num = 0
    gunman = []
    # xmin, ymin, xmax, ymax
    gunman_coords = []
    # while video is running
    while True:
        g_xmin, g_ymin, g_xmax, g_ymax = 0, 0, 0, 0
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        possible_gunman_id = []
        possible_gunman_flag_value = []
        guns = []
        people = []
        gunman_coords_prev = gunman_coords
        gunman_coords = []
        most_flags = 0
        # wait for first frame to arrive
        while True:
            try:
                with open('./data/numpy_arrays/array{}.npy'.format(frame_num), 'rb') as f:
                    frame = np.load(f)
                    break
            except:
                continue

        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on with tensorflow
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['cup', 'person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
    
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
    
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
    
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    

            # capture boxes of all guns
            if class_name == 'cup':
                g_xmin, g_ymin, g_xmax, g_ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                guns.append(g_xmin)
                guns.append(g_ymin)
                guns.append(g_xmax)
                guns.append(g_ymax)

            # capture boxes of all people
            if class_name == 'person':
                gunman_flag = 0
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                people.append(xmin)
                people.append(ymin)
                people.append(xmax)
                people.append(ymax)
                possible_gunman_id.append(str(track.track_id))
                # save coords if is gunman
                if str(track.track_id) in gunman:
                    gunman_coords.append(xmin)
                    gunman_coords.append(ymin)
                    gunman_coords.append(xmax)
                    gunman_coords.append(ymax)
                    

        # if there is a gun, and people, wiegh the chance each person is a gunman
        if len(people) > 0 and len(guns) > 0:
                # iterates through all known gun instances and all known people instances
                for i in range(int(len(people)/4)):
                    for j in range(int(len(guns)/4)):
                      # counts number of overlapping edges between person and gun
                      if guns[0+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[1+j] >= people[1+i] and guns[3+j] <= people[3+i]:
                          gunman_flag += 4
		
                      elif guns[0+j] < people[0+i] and guns[2+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[3+j] >= people[1+i] and guns[3+j] <= people[3+i] and guns[1+j] >= people[1+i] and guns[1+j] <= people[3+i]:
                          gunman_flag += 2

                      elif guns[1+j] < people[1+i] and guns[3+j] >= people[1+i] and guns[3+j] <= people[3+i] and guns[2+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[0+j] >= people[0+i] and guns[0+j] <= people[2+i]:
                          gunman_flag += 2

                      elif guns[2+j] > people[2+i] and guns[0+j] <= people[2+i] and guns[0+j] >= people[0+i] and guns[1+j] <= people[3+i] and guns[1+j] >= people[1+i] and guns[3+j] <= people[3+i] and guns[3+j] >= people[1+i]:
                          gunman_flag += 2

                      elif guns[3+j] > people[3+i] and guns[1+j] <= people[3+i] and guns[1+j] >= people[1+i] and guns[0+j] <= people[2+i] and guns[0+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[2+j] >= people[0+i]:
                          gunman_flag += 2
                  
                      elif guns[0+j] < people[0+i] and guns[1+j] < people[1+i] and guns[2+j] <= people[2+i] and guns[3+j] >= people[1+i] and guns[3+j] <= people[3+i] and guns[1+j] >= people[1+i] and guns[1+j] <= people[3+i]:
                          gunman_flag += 1

                      elif guns[1+j] < people[1+i] and guns[0+j] < people[0+i] and guns[3+j] <= people[3+i] and guns[2+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[0+j] >= people[0+i] and guns[0+j] <= people[2+i]:
                          gunman_flag += 1

                      elif guns[2+j] > people[2+i] and guns[3+j] > people[3+i] and guns[0+j] >= people[0+i] and guns[1+j] <= people[3+i] and guns[1+j] >= people[1+i] and guns[3+j] <= people[3+i] and guns[3+j] >= people[1+i]:
                          gunman_flag += 1
 
                      elif guns[3+j] > people[3+i] and guns[2+j] > people[2+i] and guns[1+j] >= people[1+i] and guns[0+j] <= people[2+i] and guns[0+j] >= people[0+i] and guns[2+j] <= people[2+i] and guns[2+j] >= people[0+i]:
                          gunman_flag += 1

                      possible_gunman_flag_value.append(gunman_flag)

        # if there are suspects associate the gun
        if len(possible_gunman_flag_value) > 0:        
            out.write(result) 
            for i in range(len(possible_gunman_id)):
                if possible_gunman_flag_value[i] == 4 and possible_gunman_id[i] not in gunman:
                    gunman.append(possible_gunman_id[i])
                else:
                    if possible_gunman_flag_value[i] > most_flags:
                        most_flags = possible_gunman_flag_value[i]
        
            if most_flags > 0:
                for i in range(len(possible_gunman_id)):
                    if possible_gunman_flag_value[i] >= most_flags:
                        if len(gunman) == 0:
                            gunman.append(possible_gunman_id[i])
                        else:
                            if possible_gunman_id[i] not in gunman:
                                gunman.append(possible_gunman_id[i])
       
        # what direction is the gunman moving
        if len(gunman_coords) > 0 and len(gunman_coords_prev) > 0:          
            for i in range(int(len(gunman_coords)/4)):
                statement = "Suspect moving, "
                delta_xmin = gunman_coords[0+i] - gunman_coords_prev[0+i]
                delta_ymin = gunman_coords[1+i] - gunman_coords_prev[1+i]
                delta_xmax = gunman_coords[2+i] - gunman_coords_prev[2+i]
                delta_ymax = gunman_coords[3+i] - gunman_coords_prev[3+i]

                if delta_xmin < 0 and delta_xmin < 0 or delta_xmin <= 0 and delta_xmin < 0 or delta_xmin < 0 and delta_xmin <= 0:
                    statement += "left   "
                elif delta_xmin > 0 and delta_xmin > 0 or delta_xmin >= 0 and delta_xmin > 0 or delta_xmin > 0 and delta_xmin >= 0:
                    statement += "right  "
                if delta_ymin < 0 and delta_ymax < 0 or delta_ymin <= 0 and delta_ymax < 0 or delta_ymin < 0 and delta_ymax <= 0:
                    statement += "up     "
                elif delta_ymin > 0 and delta_ymax > 0 or delta_ymin >= 0 and delta_ymax > 0 or delta_ymin > 0 and delta_ymax >= 0:
                    statement += "down   "
                print("{}             ".format(statement), end='\r')
                
            
        ############################################################

        # calculate frames per second of running detections    
        fps = 1.0 / (time.time() - start_time)
        #print("Frame Completed in: %.3f ms  " % ((time.time() - start_time)*1000), end='\r')
        print("				       FPS: %.2f" % fps, end='\r')
        print("						         GUNMAN: {}".format(gunman), end='\r')
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
            

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
