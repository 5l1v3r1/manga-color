from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize
import os
from keras.applications import vgg19
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob
from skimage.io import imread


def compute_descriptor(batch):
    if not hasattr(compute_descriptor, "vgg_model"):
        compute_descriptor.vgg_model = vgg19.VGG19(include_top=False, input_shape=(None, None, 3))
    descriptors = []
    for shape in [(200, 200), (250, 250), (300, 300)]:
        batch = resize(batch, shape, preserve_range=True)
        descriptor = vgg19.preprocess_input(np.expand_dims(batch, axis=0))
        descriptors.append(compute_descriptor.vgg_model.predict(descriptor))
    return descriptors


def divide_image(img):
    col = img.shape[1]
    return img[:, (col/2):], img[:, :(col/2)]


def read_images(folder_colored, folder_gray, divide_colored=True):
    names_colored = os.listdir(folder_colored)
    names_colored.sort()
    names_gray = os.listdir(folder_gray)
    names_gray.sort()

    images_colored = []
    images_gray = []

    for name in names_colored:
        img = cv2.imread(os.path.join(folder_colored, name))
        if divide_colored:
            images_colored += list(divide_image(img))
        else:
            images_colored.append(img)

    for name in names_gray:
        img = cv2.imread(os.path.join(folder_gray, name))
        if img is None:
            img = imread(os.path.join(folder_gray, name))
            if len(img.shape) == 2:
                img = gray2rgb(img.astype('float64'))

        if not is_colored(img):
            images_gray.append(img)

    return images_colored, images_gray


def is_colored(img, th=50):
    img = img.astype('float64')
    colored_score = np.mean((gray2rgb(rgb2gray(img)) - img) ** 2)
    return colored_score > th


def compute_norm_matrix(images_colored, images_gray):
    descriptors_colored = [compute_descriptor(img) for img in images_colored]
    descriptors_gray = [compute_descriptor(img) for img in images_gray]

    norm_matrix = np.zeros([len(descriptors_colored), len(descriptors_gray)])
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            for dc, dg in zip(descriptors_colored[i], descriptors_gray[j]):
                norm_matrix[i, j] += np.sum((dc - dg) ** 2)

    return norm_matrix


def find_number_of_skips(norm_matrix):
    max_score = 0
    skip_number = None
    correspondence = np.argmin(norm_matrix, axis=0)

    for row_begin in range(norm_matrix.shape[0]):
        for col_begin in range(norm_matrix.shape[1]):
            score = 0
            num_items = min(norm_matrix.shape[0] - row_begin, norm_matrix.shape[1] - col_begin)
            for k in range(num_items):
                i = row_begin + k
                j = col_begin + k
                score += (correspondence[j] == i)

            if score >= max_score:
                max_score = score
                skip_number = (row_begin, col_begin)

    return skip_number


def align_images(images_colored, images_gray):
    norm_matrix = compute_norm_matrix(images_colored, images_gray)
    skip_number_left = find_number_of_skips(norm_matrix)
    skip_number_right = find_number_of_skips(norm_matrix[::-1, ::-1])

    images_colored = images_colored[skip_number_left[0]:]
    images_gray = images_gray[skip_number_left[1]:]

    if skip_number_right[0] != 0:
        images_colored = images_colored[:-skip_number_right[0]]
    if skip_number_right[1] != 0:
        images_gray = images_gray[:-skip_number_right[1]]

    return images_colored, images_gray


def save_images(images_colored, images_gray, out_folder, chapter_name):
    out_colored = os.path.join(out_folder, chapter_name, 'colored')
    out_gray = os.path.join(out_folder, chapter_name, 'gray')

    if not os.path.exists(out_colored):
        os.makedirs(out_colored)
    if not os.path.exists(out_gray):
        os.makedirs(out_gray)

    def save_image_list(folder, image_list):
        for i, img in enumerate(image_list):
            name = str(i).zfill(3) + '.jpg'
            cv2.imwrite(os.path.join(folder, name), img)

    save_image_list(out_colored, images_colored)
    save_image_list(out_gray, images_gray)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='/media/gin/data/manga-dataset/love-ru',
                        help="Output directory with aligned result")
    parser.add_argument("--input_dir_colored_pattern",
                        default='/media/gin/data/share/To Love-Ru Darkness - Digital Colored Comics/%s',
                        help="Input dir pattern (%s will be replaced with number) for colored images")

    parser.add_argument("--input_dir_gray_pattern",
                        default='/media/gin/data/share/To LOVE-RU Darkness/To LOVE-RU Darkness %s *',
                        help="Input dir pattern (%s will be replaced with number) for gray images")

    parser.add_argument("--start_from", default=1, type=int, help='First folder number')
    parser.add_argument("--end_at", type=int, default=31, help='Last folder number')
    parser.add_argument("--divide_colored", type=int, default=False)

    args = parser.parse_args()

    for i in tqdm(range(args.start_from, args.end_at + 1)):
        index_as_str = str(i).zfill(3)
        images_colored, images_gray = read_images(glob.glob(args.input_dir_colored_pattern % index_as_str)[0],
                                                  glob.glob(args.input_dir_gray_pattern % index_as_str)[0],
                                                  divide_colored=args.divide_colored)

        images_colored, images_gray = align_images(images_colored, images_gray)
        save_images(images_colored, images_gray, args.output_dir, index_as_str)


if __name__ == "__main__":
    main()
