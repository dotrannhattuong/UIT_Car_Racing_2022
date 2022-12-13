

COCO_CLASSES_LIST = [
    'camtrai',
    'camphai',
    'camthang',
    'trai',
    'phai',
    'thang',
    'None'
]


def get_cls_dict(category_num):
    if category_num == 6:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
