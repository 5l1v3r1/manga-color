import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict

import argparse
from skimage.io import imread, imsave
from detector import TextDetector

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default='bubble-annotation.csv',
                    help="File with annotations from http://www.robots.ox.ac.uk/~vgg/software/via/via.html")
parser.add_argument("--input_images", default='images',
                    help='Folder with images')
parser.add_argument("--output_dir_annotations", default='anotations-icdar',
                    help="Directory with annotations in icdar format")
parser.add_argument("--output_dir_images", default='images-icdar',
                    help="Directory with divided images")

args = parser.parse_args()

df = pd.read_csv(args.input_csv)


def to_bbox(region_shape_attributes):
    dict_rs = json.loads(region_shape_attributes)
    bbox = np.zeros(4)
    bbox[0] = dict_rs[u'x']
    bbox[1] = dict_rs[u'y']
    bbox[2] = dict_rs[u'x'] + dict_rs[u'width']
    bbox[3] = dict_rs[u'y'] + dict_rs[u'height']
    return bbox


def cut_bbox(bbox, bounds):
    bbox = bbox.copy()
    if bounds[0] > bbox[2] or bounds[2] < bbox[0]:
        return None
    if bounds[1] > bbox[3] or bounds[3] < bbox[1]:
        return None
    bbox[0] = max(bbox[0], bounds[0]) - bounds[0]
    bbox[1] = max(bbox[1], bounds[1]) - bounds[1]
    bbox[2] = min(bbox[2], bounds[2]) - bounds[0]
    bbox[3] = min(bbox[3], bounds[3]) - bounds[1]
    return bbox

dict_with_annotations = defaultdict(list)

for i, row in df.iterrows():
    name = row['#filename']
    region_shape_attributes = row['region_shape_attributes']
    if region_shape_attributes == '{}':
        continue
    bbox = to_bbox(region_shape_attributes)
    dict_with_annotations[name].append(bbox)


if not os.path.exists(args.output_dir_annotations):
    os.makedirs(args.output_dir_annotations)

if not os.path.exists(args.output_dir_images):
    os.makedirs(args.output_dir_images)


new_dict_with_annotations = defaultdict(list)
for key, value in dict_with_annotations.iteritems():
    bboxes = np.array(value)
    img = imread(os.path.join(args.input_images, key))
    #Split images in 4 parts
    x_step = img.shape[1]/2
    y_step = img.shape[0]/2

    for i, x_lower in enumerate(range(0, img.shape[1] - 1, x_step)):
        for j, y_lower in enumerate(range(0, img.shape[0] - 1, y_step)):
            new_img_name = key.replace('.jpg', '') + '_' + str(i) + str(j) + '.jpg'
            new_img = img[y_lower:(y_lower + y_step), x_lower:(x_lower + x_step)]
            for bbox in bboxes:
                new_bbox = cut_bbox(bbox, np.array([x_lower, y_lower, x_lower + x_step, y_lower + y_step]))
                if new_bbox is not None:
                    new_dict_with_annotations[new_img_name].append(new_bbox)

            imsave(os.path.join(args.output_dir_images, new_img_name), new_img)
            # TextDetector.plot_predictions(new_img, np.array(new_dict_with_annotations[new_img_name]))

dict_with_annotations = new_dict_with_annotations


for key, value in dict_with_annotations.iteritems():
    with open(os.path.join(args.output_dir_annotations, 'gt_' + key.replace('.jpg', '') + '.txt'), 'w') as f:
        for v in value:
            print >>f, ','.join(map(str, map(int, v))) + ',""'
