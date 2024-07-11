#!/usr/bin/env python3

import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from threading import Lock, Thread
from queue import Queue
from time import sleep


lock = Lock()
frame_queue = Queue(maxsize=10)
run_signal = False
exit_signal = False


def torch_thread(weights, img_size, conf_thres=0.5, iou_thres=0.45):
    global exit_signal, run_signal, annotator, frame_queue

    print("Intializing Model...")

    model = YOLO(weights)
    
    while not exit_signal:
        if run_signal:
            lock.acquire()
            image_net, point_cloud = frame_queue.get()
            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)

            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)
            image = det[0].orig_img.copy()
            annotator = Annotator(image)

            for r in det:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Use the annotator to get the color of the current box
                    class_id = int(box.cls)
                    color = colors(class_id, True)

                    conf = box.conf.item()
                    class_label = model.names[class_id]
                    label = f"{class_label} {conf:.2f}"

                    # Annotator the bounding box
                    annotator.box_label([x1, y1, x2, y2], label=label, color=color)
                    cv2.circle(annotator.im, center, 4, color, -1)

                    # Retrieve point cloud data for the bounding box center
                    _, pc_value = point_cloud.get_value(center[0], center[1])
                    x, y, z = pc_value[0], pc_value[1], pc_value[2]
                    xyz_label = f"X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                    cv2.putText(annotator.im, xyz_label, (x1 - 150, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            lock.release()
            run_signal = False
        sleep(0.01)
    

def main(
        weights='runs/train/duo_land_exp/weights/best.pt',
        svo=None,
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.45,
        save='mydata/experiment/video/output.mp4',
):
    global exit_signal, run_signal, annotator, frame_queue
    
    capture_thread = Thread(target=torch_thread, kwargs={'weights': weights, 'img_size': img_size, 
                                                         "conf_thres": conf_thres, "iou_thres": iou_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()

    if svo is not None:
        input_type.set_from_svo_file(svo)

    # Create a InitParameters object
    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    if svo is None:
        init_params.camera_resolution = sl.RESOLUTION.HD1080

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left = sl.Mat()
    print("🆗Initialized")

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    point_cloud = sl.Mat(camera_res.width, camera_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fps = camera_infos.camera_configuration.fps
    fps = 20
    out = cv2.VideoWriter(save, fourcc, fps, (camera_res.width, camera_res.height))

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Get the image
            lock.acquire()
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, camera_res)
            image_net = image_left.get_data()
            frame_queue.put((image_net, point_cloud))
            lock.release()
            run_signal = True

            # Detection running on the other thread
            while run_signal:
                sleep(0.01)

            # Display
            lock.acquire()
            cv2.imshow("ZED", annotator.im)
            out.write(annotator.im)
            lock.release()
            key = cv2.waitKey(1)

            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    cv2.destroyAllWindows()
    zed.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/duo_land_exp/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save', type=str, default='mydata/experiment/video/output.mp4', help='output video file')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    with torch.no_grad():
        main(**vars(opt))
