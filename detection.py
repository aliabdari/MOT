import numpy as np
import torch

from utils.torch_utils import time_sync
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return bbox_left, bbox_top, x_c, y_c, w, h

def detect_reid(frame, model, stride, imgsz, half, device, deepsort, opt):
    boxes = []
    classes = []
    track_classes = []
    identities = []
    bbox_xywh = []
    confs = []

    img = letterbox(frame, imgsz, stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_sync()
    pred = model(img, augment=opt.augment)[0]

    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    t3 = time_sync()

    # this is for supporting multiple frame push
    # here function return is assumng one frame push (output of each frame should be added to a global list and that list should be used for function return)
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], frame.shape).round()

            # Adapt detections to deep sort input format
            for *xyxy, conf, cls in det:
                x_l, y_l, x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                obj = [x_c, y_c, bbox_w, bbox_h]
                # boxes.append(np.array([x_l,y_l,bbox_w, bbox_h]))
                bbox_xywh.append(obj)
                confs.append([conf.item()])
                classes.append(cls.item())
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)
            # Pass detections to deepsort
            outputs = deepsort.update(xywhs, confss, classes, frame)
            # draw boxes for visualization
            if len(outputs) > 0:
                boxes = outputs[:, :4].tolist()
                identities = outputs[:, -2]
                track_classes = outputs[:, -1].tolist()
                # draw_boxes(frame, bbox_xyxy, identities)
                identities = identities.tolist()
        else:
            deepsort.increment_ages()
    t2 = time_sync()

    # Print time (inference + NMS)
    # print('Done. (%.3fs)' % (t2 - t1))
    return boxes, identities, track_classes
