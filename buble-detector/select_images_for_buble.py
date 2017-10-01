import numpy as np
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='/media/gin/data/manga-dataset/',
                    help="Directory with images")
parser.add_argument("--output_dir", default='images',
                    help="Directory with selected images")
args = parser.parse_args()

image_count = {'one-piece': 100, 'bleach': 5, 'death-note': 10, 'love-ru': 20, 'naruto': 25}
np.random.seed(0)
selected_images = []
for key, value in image_count.iteritems():
    images = []
    for root, _, files in os.walk(os.path.join(args.input_dir, key)):
        images += [os.path.join(root, name) for name in files]
    images = np.array(images)
    selected_images += list(np.random.choice(images, size=value))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for i, name in enumerate(selected_images):
    shutil.copy(name, os.path.join(args.output_dir, str(i).zfill(3)) + '.jpg')
