import argparse
import cv2
import numpy as np
import os
from kafka import KafkaConsumer, KafkaProducer, errors
import json
import time


def diff_frame(frame1, frame2, color_threshold):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gaussian_frame1 = cv2.GaussianBlur(gray_frame1, (3, 3), 0)
    gaussian_frame2 = cv2.GaussianBlur(gray_frame2, (3, 3), 0)
    dframe = cv2.absdiff(gaussian_frame1, gaussian_frame2)
    ret, diff = cv2.threshold(dframe, color_threshold, 255, cv2.THRESH_BINARY)
    return diff


def intrusion_detection(source, output_dir, background, interval, region, color_threshold, pixel_threshold, kafka_server, intrusion_topic, alarm_topic, cd_step):
    kafka_producer = KafkaProducer(bootstrap_servers=kafka_server,
                                   value_serializer=lambda m:
                                   json.dumps(m).encode('utf-8'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    roi = region.replace('\n', '').split(',')
    for i in range(len(roi)):
        roi[i] = int(roi[i])

    background_frame = cv2.imread(background)[roi[1]:roi[3], roi[0]:roi[2]]

    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise RuntimeError("Could not open the video url")

    print("Sucessfully opened video file")

    step = 0
    has_alarm = False
    last_step = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("No frame read. Stopping.")
            break
        if step % interval == 0:
            diff = diff_frame(background_frame, frame[roi[1]:roi[3], roi[0]:roi[2]], color_threshold)
            diff_num = np.sum(diff == 255)
            if diff_num > pixel_threshold:
                frame_path = os.path.join(output_dir, "frame-{:d}-{:d}.jpg".format(step, diff_num))
                cv2.imwrite(frame_path, frame)
                msg = {"path": frame_path}
                kafka_producer.send(intrusion_topic, msg)
                last_step = step
                if not has_alarm:
                    alarm_msg = {"type": "intrusion-start", "step": step, "snapshot": frame_path}
                    kafka_producer.send(alarm_topic, alarm_msg)
                    has_alarm = True
            else:
                if has_alarm and step - last_step >= cd_step:
                    frame_path = os.path.join(output_dir, "frame-{:d}-{:d}.jpg".format(step, diff_num))
                    cv2.imwrite(frame_path, frame)
                    alarm_msg = {"type": "intrusion-end", "step": step, "snapshot": frame_path}
                    kafka_producer.send(alarm_topic, alarm_msg)
                    has_alarm = False
        step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source name, like video file or live source",
                        default="source")
    parser.add_argument("-o", "--output", help="output file dir",
                        default="output")
    parser.add_argument("-b", "--background", help="background frame",
                        default="backgroud.jpg")
    parser.add_argument("-i", "--interval", help="fps",
                        default="10")
    parser.add_argument("-r", "--region", help="region or interest's coordination: xmin,ymin,xmax,ymax",
                        default="108,192,216,384")
    parser.add_argument("-c", "--color", help="color diff threshold",
                        default="10")
    parser.add_argument("-p", "--pixel", help="diff pixel num threshold",
                        default="10")
    parser.add_argument("-ks", "--kafka_server", help="kafka server",
                        default="172.16.29.105:9092")
    parser.add_argument("-it", "--intrusion_topic", help="intrusion topic",
                        default="intrusion-detection")
    parser.add_argument("-at", "--alarm_topic", help="alarm topic",
                        default="alarm-warning")
    parser.add_argument("-cd", "--cd_step", help="alarm cd step",
                        default="30")
    args = parser.parse_args()
    intrusion_detection(args.source, args.output, args.background, int(args.interval), args.region, int(args.color), int(args.pixel), args.kafka_server, args.intrusion_topic, args.alarm_topic, int(args.cd_step))


if __name__ == "__main__":
    main()
