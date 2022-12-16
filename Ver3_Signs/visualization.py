from utils.yolo_classes import COCO_CLASSES_LIST

import numpy as np
import cv2


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: tổng số class

    # Output
      bgrs: a list of (B, G, R) tuples
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """

    # Arguments
      img: the input image as a numpy array.
      text: the text để vẽ.
      topleft: góc trái của boudingbox để vẽ.
      color: color of the patch.

   
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])
    h = min(h, img_h - topleft[1])

    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: danh sách tên các class.
    """

    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))

    def draw_bboxes(self, img, boxes, confs, clss, session, inputname):
        """vẽ bounding boxes."""
        area = 0
        cl = -1
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]

            if (x_max+x_min)//2 < 320:
              continue

            X_test = img[y_min:y_max, x_min:x_max]
            X_test = cv2.resize(X_test, (30, 30))
            X_test = X_test.astype('float32')/255
            X_test = X_test.reshape(1, 30, 30, 3)

            prediction = session.run(None, {inputname: X_test})
            
            prediction = np.squeeze(prediction)
            cl = np.argmax(prediction)
            area = (x_max-x_min)*(y_max-y_min)

        # lst_class[1:] = lst_class[0:-1]
        # lst_class[0] = cl
        # cl = int(most_frequent(list(lst_class)))
        cl = COCO_CLASSES_LIST[cl]
        return img, cl, area

lst_class = np.zeros(5) - 1

def most_frequent(List):
    return max(set(List), key = List.count)
