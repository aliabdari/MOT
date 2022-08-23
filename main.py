import cv2
import torch
import os

from detection import detect_reid
from utils.torch_utils import select_device
from models.experimental import attempt_load

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import argparse

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str,
                    default='./trained_models/Monkey_detector2.pt', help='model.pt path(s)')
# file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640,
                    help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float,
                    default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.25,
                    help='IOU threshold for NMS')  # this is for detection nms part
parser.add_argument('--device', default='',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true',
                    help='display results')
parser.add_argument('--classes', nargs='+', type=int,
                    help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true',
                    help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true',
                    help='augmented inference')

#########################################################
# all this flags transfered to config_deepsort file flag
#########################################################
parser.add_argument("--max_age",
                    help="Maximum number of frames to keep alive a track without associated detections.",
                    type=int, default=30)
parser.add_argument("--min_hits",
                    help="Minimum number of associated detections before track is initialised.",
                    type=int, default=3)
parser.add_argument("--iou_threshold", help="Minimum IOU for match.",
                    type=float, default=0.3)  # this is for sort matching part
parser.add_argument(
    "--min_detection_height", help="Threshold on the detection bounding "
                                   "box height. Detections with height smaller than this value are "
                                   "disregarded", default=0, type=int)
parser.add_argument(
    "--max_cosine_distance", help="Gating threshold for cosine distance "
                                  "metric (object appearance).", type=float, default=0.2)
parser.add_argument(
    "--nn_budget", help="Maximum size of the appearance descriptors "
                        "gallery. If None, no budget is enforced.", type=int, default=None)
parser.add_argument(
    "--model",
    default="resources/networks/mars-small128.pb",
    help="Path to freezed inference graph protobuf.")
#########################################################################

parser.add_argument("--config_deepsort", type=str,
                    default="deep_sort_pytorch/configs/deep_sort.yaml")

opt = parser.parse_args()
args = vars(opt)

colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (125, 128, 0), (0, 128, 128), (128, 0, 128)]

input_video = './videos/VID2.mp4'
output_video = './videos/VID2_output.mp4'
result_text = './videos/results.txt'

if os.path.exists(result_text):
    os.remove(result_text)
data_file = open(result_text, "w")

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)  # input video FPS
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (W, H))

total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 1

weights, view_img, imgsz = opt.weights, opt.view_img, opt.img_size
device = select_device(opt.device)
half = device.type != 'cpu'
model = attempt_load(weights)
stride = int(model.stride.max())

cfg = get_config()
cfg.merge_from_file(args["config_deepsort"])
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

with torch.no_grad():
    while True:
        success, image = cap.read()
        print("frame " + str(frame_number) +
              " of " + str(total_number_of_frames))
        frame_number += 1
        if not success:
            break

        # Detection Stage #
        boxes, ids, classes = detect_reid(image, model, stride, imgsz, half, device, deepsort, opt)
        # print(boxes)
        for b in boxes:
            x_min = b[0]
            y_min = b[1]
            x_max = b[0] + b[2]
            y_max = b[1] + b[3]

            start_point = (x_min, y_min)
            end_point = (x_max, y_max)

            index = boxes.index(b)
            image = cv2.rectangle(image, start_point, end_point, colors[index % len(colors)], 4)
            cv2.putText(image, str(ids[index]), (x_min - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 255, 0), 4)
            print(str(frame_number) + " " + str(ids[index]) + " " + str(b))
            data_file.write(str(frame_number) + "," + str(ids[index]) + ","
                            + str(b[0]) + "," + str(b[1]) + "," + str(b[2]) + "," + str(b[3]))
            data_file.write("\n")

        cv2.putText(image, str(frame_number), (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 0), 4)
        out.write(image)
        # print(ids)
    out.release()
    data_file.close()
