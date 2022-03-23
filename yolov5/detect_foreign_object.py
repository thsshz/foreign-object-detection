import argparse
import os
import sys
from kafka import KafkaConsumer, KafkaProducer, errors
import json

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
def run(weights, save_dir, kafka_server, kafka_topic):
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.25   # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''
    data = 'yolov5/data/coco128.yaml'
    bs = 1
    augment = False  # augmented inference
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    visualize = False  # visualize features
    line_thickness = 3  # bounding box thickness (pixels)
    save_img = True
    kafka_sink = True
    out_topic = "foreign-object-detection"
    if kafka_sink:
        kafka_producer = KafkaProducer(bootstrap_servers=kafka_server,
                                       value_serializer=lambda m:
                                       json.dumps(m).encode('utf-8'))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    kafka_consumer = KafkaConsumer(kafka_topic, bootstrap_servers=kafka_server)
    while True:
        raw_messages = kafka_consumer.poll(timeout_ms=1000.0, max_records=5000)
        for _, msg_list in raw_messages.items():
            for msg in msg_list:
                msg_value = json.loads(msg.value.decode('utf-8'))
                image_path = str(msg_value["path"])
                dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=pt)
                dt, seen = [0.0, 0.0, 0.0], 0
                for path, im, im0s, vid_cap, s in dataset:
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    pred = model(im, augment=augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    dt[2] += time_sync() - t3

                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        save_path = os.path.join(save_dir, p.split('/')[-1])
                        s += '%gx%g ' % im.shape[2:]  # print string
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        bboxes = []
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                if kafka_sink:
                                    bbox = {
                                        "label": names[c],
                                        "score": float(conf),
                                        "topleftx": int(xyxy[0]),
                                        "toplefty": int(xyxy[1]),
                                        "bottomrightx": int(xyxy[2]),
                                        "bottomrighty": int(xyxy[3])
                                    }
                                    bboxes.append(bbox)
                                if save_img:  # Add bbox to image
                                    label = f'{names[c]} {conf:.2f}'
                                    annotator.box_label(xyxy, label, color=colors(c, True))

                        # Stream results
                        im0 = annotator.result()

                        # Save results (image with detections)
                        if save_img:
                            cv2.imwrite(save_path, im0)

                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                LOGGER.info(
                    f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                if kafka_sink:
                    msg = {"frame": image_path,
                           "bbox": bboxes}
                    kafka_producer.send(out_topic, msg)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", '--weights', help='model path',
                        default='yolov5s.pt')
    parser.add_argument("-s", '--save_dir', help='save dir path',
                        default='detect_results')
    parser.add_argument("-ks", "--kafka_server", help="kafka server",
                        default="172.16.29.105:9092")
    parser.add_argument("-kt", "--kafka_topic", help="kafka topic",
                        default="intrusion-detection")
    opt = parser.parse_args()
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
