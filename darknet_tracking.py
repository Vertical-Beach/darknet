from ctypes import *
import argparse
import json
import os
import random
import time
from threading import Thread, enumerate
from queue import Queue

import cv2
import darknet
import sort
import numpy as np

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--out_filename2", type=str, default="",
                        help="tracking video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame, frame_queue, darknet_image_queue):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                interpolation=cv2.INTER_LINEAR)
    frame_queue.put(frame)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    darknet_image_queue.put(img_for_detect)


def inference(darknet_image_queue, detections_queue, fps_queue, tracked_queue, pedestrian_tracker, car_tracker):
    def convert(bbox, conf, classid):
        left_x, top_y, width, height = bbox
        right_x = left_x + width
        bottom_y = top_y + height
        # dummy = 9999999 #?
        dummy = 0 #?
        return [left_x, top_y, right_x, bottom_y, conf/100, dummy, classid]
    
    def convert2(bbox):
        left_x, top_y, right_x, bottom_y = bbox
        width = right_x - left_x
        height = bottom_y - top_y
        return left_x, top_y, width, height

    def tracking(pedestrian_tracker, car_tracker, detections):
        ped_bboxes = []
        car_bboxes = []
        for label, conf, bbox in detections:
            if label == 'pedestrian':
                ped_bboxes.append(convert(bbox, float(conf), 0))
            elif label == 'car':
                car_bboxes.append(convert(bbox, float(conf), 1))
        ped_tracked_objects = pedestrian_tracker.update(np.array(ped_bboxes))
        car_tracked_objects = car_tracker.update(np.array(car_bboxes))
        tracked_objects = []
        print(ped_tracked_objects.shape)
        for tracked_ped in ped_tracked_objects:
            obj_id = tracked_ped[4]
            tracked_objects.append(('Pedestrian', obj_id, convert2(tracked_ped[0:4])))

        for tracked_car in car_tracked_objects:
            obj_id = tracked_car[4]
            tracked_objects.append(('Car', obj_id, convert2(tracked_car[0:4])))
        
        return tracked_objects

    darknet_image = darknet_image_queue.get()
    prev_time = time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
    tracked_objects = tracking(pedestrian_tracker, car_tracker, detections)
    detections_queue.put(detections)
    tracked_queue.put(tracked_objects)
    fps = int(1/(time.time() - prev_time))
    fps_queue.put(fps)
    print("FPS: {}".format(fps))
    darknet.print_detections(detections, args.ext_output)
    darknet.free_image(darknet_image)


def drawing(frame_queue, detections_queue, fps_queue, tracked_queue):
    def draw_tracked_boxes(detections, image):
        colors = [
            [255, 71, 71],
            [235, 142, 71],
            [235, 210, 71],
            [188, 235, 71],
            [118, 235, 71],
            [71, 235, 96],
            [71, 235, 164],
            [71, 235, 235],
            [71, 96, 235],
            [118, 71, 235],
            [188, 71, 235],
            [235, 71, 210],
            [235, 71, 142]
        ]
        import cv2
        for label, obj_id, bbox in detections:
            color = colors[obj_id % len(colors)]
            # color.reverse() #RGB -> BGR
            left, top, right, bottom = darknet.bbox2points(bbox)
            cv2.rectangle(image, (left, top), (right, bottom), color, 3)
            cv2.putText(image, "{} [id={}, {}%]".format(label, obj_id, confidence),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)
        return image

    random.seed(3)  # deterministic bbox colors
    
    def bbox_to_dict(frame2, tracked_objects):
        res = {}
        for label, obj_id, bbox in tracked_objects:
            x1, y1, w, h = convert2original(frame2, bbox)
            x2 = x1 + w
            y2 = x2 + h
            if label not in res:
                res[label] = []
            res[label].append({
                "id": int(obj_id),
                "box2d": [x1, y1, x2, y2]    
            })
        return res

    frame = frame_queue.get()
    detections = detections_queue.get()
    tracked_objects = tracked_queue.get()
    frame_result = None

    fps = fps_queue.get()
    detections_adjusted = []
    tracked_adjusted = []
    if frame is not None:
        frame2 = frame.copy()

        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(frame, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)

        for label, obj_id, bbox in tracked_objects:
            bbox_adjusted = convert2original(frame2, bbox)
            tracked_adjusted.append((str(label), int(obj_id), bbox_adjusted))
        frame_result = bbox_to_dict(frame2, tracked_objects)
        image2 = draw_tracked_boxes(tracked_adjusted, frame2)

        image_v = cv2.vconcat([image, image2])

        if not args.dont_show:
            resized = cv2.resize(image_v, None, fx=0.5, fy=0.5)
            cv2.imshow('Inference/Tracking', resized)
            cv2.waitKey(1)
    return frame_result, image_v


if __name__ == '__main__':

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    tracked_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    json_filename = os.path.basename(args.input.replace('.mp4', '.json'))
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*2
    
    frame_results = []
    video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    pedestrian_tracker = sort.Sort()
    car_tracker = sort.Sort()
    
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        video_capture(frame, frame_queue, darknet_image_queue)
        inference(darknet_image_queue, detections_queue, fps_queue, tracked_queue, pedestrian_tracker, car_tracker)
        frame_result, image_v = drawing(frame_queue, detections_queue, fps_queue, tracked_queue)
        frame_results.append(frame_result)
        
        if args.out_filename is not None:
            video.write(image_v)
    
    open(json_filename, 'w').write(json.dumps(frame_results))
    cap.release()
    video.release()
    cv2.destroyAllWindows()