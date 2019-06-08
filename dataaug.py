from PIL import Image
import numpy as np


def paste_data(train_exp, img_infos, norm_path, anno_info, img_ids, star_id, rand_sample):
    norm_img = Image.open(norm_path)
    norm_array = np.array(norm_img)
    norm_w, norm_h = norm_img.size
    norm_area = norm_w * norm_h
    norm_name = norm_path.split('/')[-1]
    mask_zero = np.zeros((norm_h, norm_w, 1), np.uint8)
    annos = []
    bboxs = []
    for i in range(rand_sample):
        bbox_id = star_id + i
        while True:
            rnd = np.random.randint(len(anno_info))

            bbox_info = anno_info[rnd]
            anno = bbox_info.copy()
            img_id = bbox_info['image_id']
            img_list = img_infos[img_id]

            img_name = img_list['file_name']
            img_path = os.path.join(train_exp, 'restricted', img_name)
            img = Image.open(img_path)
            image = np.array(img)
            # image = cv2.imread(img_path)
            new_seg = []
            seg = anno['segmentation']

            for i in range(0, len(seg[0]), 2):
                if (seg[0][i], seg[0][i + 1]) not in new_seg:
                    new_seg.append((seg[0][i], seg[0][i + 1]))

            if Polygon(new_seg).is_valid == False:
                continue
            aug_seg, aug_bbox, aug_img = get_aug_image(new_seg, image)
            if aug_seg == None:
                continue
            _, _, bbox_w, bbox_h = aug_bbox

            bbox_area = bbox_w * bbox_h
            if bbox_area / norm_area < 0.1:
                break

        seg = aug_seg

        # bbox = anno['bbox']
        [x, y, w, h] = aug_bbox
        w_crop, h_crop = w, h
        img_id = img_ids[norm_name]
        anno['image_id'] = img_id
        anno['id'] = bbox_id

        # img = Image.open(img_path)
        img = Image.fromarray(aug_img)
        #   img.save('./work_dirs/rot/'+str(i)+'_'+norm_name)
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)
        img_draw = ImageDraw.Draw(img)

        img_mask_array = np.zeros((img_h, img_w), dtype=np.uint8)

        img_mask = Image.fromarray(img_mask_array)
        draw_mask = ImageDraw.Draw(img_mask)

        if w_crop <= 0 or h_crop <= 0:
            continue
        range_w = norm_w - w_crop
        range_h = norm_h - h_crop
        if range_w <= 0 or range_h <= 0:
            continue
        while True:
            rand_x = np.random.randint(0, range_w)
            rand_y = np.random.randint(0, range_h)
            ref_bbox = [rand_x, rand_y, w_crop, h_crop]
            judge_flag = judge(ref_bbox, bboxs)
            #     print((w_crop,h_crop))
            if judge_flag:
                break
        bboxs.append([rand_x, rand_y, w_crop, h_crop])
        anno['bbox'] = [rand_x, rand_y, w_crop, h_crop]
        new_segs = []
        new_pts = []
        for i in range(0, len(seg[0]), 2):
            new_segs.append(seg[0][i] - x + rand_x)
            new_segs.append(seg[0][i + 1] - y + rand_y)
            new_pts.append((seg[0][i], seg[0][i + 1]))
        anno['segmentation'] = [new_segs]

        draw_mask.polygon(new_pts, fill=(1), outline=(1))
        img_crop = img.crop([x, y, x + w, y + h])
        img_crop_mask = img_mask.crop([x, y, x + w, y + h])
        img_crop_array = np.array(img_crop_mask)[:, :, np.newaxis] * np.array(img_crop)

        norm_crop_img = norm_img.crop([rand_x, rand_y, rand_x + w_crop, rand_y + h_crop])
        img_past_array = (1 - np.array(img_crop_mask)[:, :, np.newaxis]) * np.array(norm_crop_img)
        img_norn_inter_array = (np.array(img_crop_mask)[:, :, np.newaxis]) * np.array(norm_crop_img)
        new_img_array = 0.9 * img_crop_array + img_past_array + 0.1 * img_norn_inter_array
        # new_img_array = np.array(img_crop)
        norm_array[rand_y:rand_y + h_crop, rand_x:rand_x + w_crop, :] = new_img_array
        annos.append(anno)
    return norm_array, annos