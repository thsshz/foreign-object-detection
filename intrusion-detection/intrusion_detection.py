import argparse
import cv2
import numpy as np
import os
from kafka import KafkaConsumer, KafkaProducer, errors
import json


def diff_frame(frame1, frame2, color_threshold):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gaussian_frame1 = cv2.GaussianBlur(gray_frame1, (3, 3), 0)
    gaussian_frame2 = cv2.GaussianBlur(gray_frame2, (3, 3), 0)
    dframe = cv2.absdiff(gaussian_frame1, gaussian_frame2)
    ret, diff = cv2.threshold(dframe, color_threshold, 255, cv2.THRESH_BINARY)
    return diff


def intrusion_detection(video_url, output_dir, start_step, interval, region, color_threshold, pixel_threshold, kafka_server, kafka_topic):
    kafka_producer = KafkaProducer(bootstrap_servers=kafka_server,
                                   value_serializer=lambda m:
                                   json.dumps(m).encode('utf-8'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video = cv2.VideoCapture(video_url)
    roi = region.replace('\n', '').split(',')
    for i in range(len(roi)):
        roi[i] = int(roi[i])
    if not video.isOpened():
        raise RuntimeError("Could not open the video url")

    print("Sucessfully opened video file")

    step = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("No frame read. Stopping.")
            break
        if step < start_step:
            step += 1
            continue
        elif step == start_step:
            last_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
        elif step % interval == 0:
            diff = diff_frame(last_frame, frame[roi[1]:roi[3], roi[0]:roi[2]], color_threshold)
            diff_num = np.sum(diff == 255)
            if diff_num > pixel_threshold:
                frame_path = os.path.join(output_dir, "frame-{:d}-{:d}.jpg".format(step, diff_num))
                cv2.imwrite(frame_path, frame)
                msg = {"path": frame_path}
                kafka_producer.send(kafka_topic, msg)
        step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="video file name",
                        default="video")
    parser.add_argument("-o", "--output", help="output file dir",
                        default="output")
    parser.add_argument("-s", "--start", help="start frame number",
                        default="10")
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
    parser.add_argument("-kt", "--kafka_topic", help="kafka topic",
                        default="intrusion-detection")
    args = parser.parse_args()
    intrusion_detection(args.video, args.output, int(args.start), int(args.interval), args.region, int(args.color), int(args.pixel), args.kafka_server, args.kafka_topic)


if __name__ == "__main__":
    main()
