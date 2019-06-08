import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from tqdm import tqdm
from mmdet.apis import inference_detector, show_result
import numpy as np
import pycocotools.mask as maskUtils
cfg = mmcv.Config.fromfile('configs/cascade_mask_rcnn_x101_32x4d_fpn_1x.py')
cfg.model.pretrained = None
import json
# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, '/home/lyk/mmdetection/work_dirs/cascade_mask_rcnn_x101_fpn_1x/epoch_32.pth')

# test a single image
# result = inference_detector(model, img, cfg)
submits_dict = dict()
json_p = "./submitxt.json"


def make_submit(image_name, preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit = dict()
    submit['image_name'] = image_name
    submit['size'] = (preds.shape[1], preds.shape[2])  # (height,width)
    submit['mask'] = dict()

    for cls_id in range(0, 5):  # 5 classes in this competition

        mask = preds[cls_id, :, :]
        cls_id_str = str(cls_id + 1)  # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        rle = maskUtils.encode(
            fortran_mask)  # encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        submit['mask'][cls_id_str] = rle

    return submit


def dump_2_json(submits, save_p):
    '''

    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8')
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()
# test a list of images
import os
import matplotlib.pyplot as plt
imgs_path = []
imgs = []
path = '/home/lyk/mmdetection/jinnan2_round2_test_b_20190424/'
for img_m in tqdm(os.listdir(path)):
    imgs_path=os.path.join(path,img_m)
    img = mmcv.imread(imgs_path)
    im_size = img.shape
    print(im_size)
    result = inference_detector(model, img, cfg)
    mk = np.zeros((5, im_size[0], im_size[1]), dtype=np.uint8)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    # segms = mmcv.concat_list(segm_result)
    segms = segm_result
    # bboxes = np.vstack(bbox_result)
    bboxes = bbox_result
    for k in range(5):
        m = np.zeros((im_size[0], im_size[1]), dtype=np.uint8)
        inds = np.where(bboxes[k][:, -1] > 0.05)[0]
        for i in inds:
            mask = maskUtils.decode(segms[k][i]).astype(np.bool)
            m += mask
        mk[k, :, :] = np.array((m > 0), dtype=np.uint8)
    final_mask = np.zeros((im_size[0], im_size[1]), np.uint8)
    for i in range(0, 5):
        final_mask += final_mask + mk[i, :, :] * (50 * (i))
    plt.title(img_m)
    plt.imshow(final_mask)
    plt.show()
    submit = make_submit(img_m, mk)
    submits_dict[img_m] = submit

dump_2_json(submits_dict, json_p)