import numpy as np

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


class TextDetector(object):
    def __init__(self, model_def ='deploy.prototxt', model_weights='VGG_scenetext_SSD_300x300_iter_60000.caffemodel'):
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order

        # set net to batch size of 1
        self.img_side = 300
        self.net.blobs['data'].reshape(1,3,self.img_side,self.img_side)

    def get_predictions_image_part(self, image, conf_th=0.25):
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']
        det_conf = detections[0,0,:,2]
        det_bboxes = detections[0,0,:,3:7]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_th]

        top_bboxes = det_bboxes[top_indices]

        top_bboxes[:, 0::2] = np.round(top_bboxes[:, 0::2] * image.shape[1]).astype('int')
        top_bboxes[:, 1::2] = np.round(top_bboxes[:, 1::2] * image.shape[0]).astype('int')

        return top_bboxes

    def get_predictions_full_image(self, image):
        image_with_pad_size = (int(np.ceil(image.shape[0] / float(self.img_side)) * self.img_side),
                               int(np.ceil(image.shape[1] / float(self.img_side)) * self.img_side))
        images_paded = image.copy()
        images_paded.resize(image_with_pad_size)

        bboxes = None
        for i in range(0, images_paded.shape[0], self.img_side):
            for j in range(0, images_paded.shape[1], self.img_side):
                i_end = i + self.img_side
                j_end = j + self.img_side
                if bboxes is None:
                    bboxes = self.get_predictions_image_part(image[i:i_end, j:j_end])
                else:
                    new_bboxes = self.get_predictions_image_part(image[i:i_end, j:j_end])
                    print (new_bboxes)
                    new_bboxes[:, 0::2] = new_bboxes[:, 0::2] + j
                    new_bboxes[:, 1::2] = new_bboxes[:, 1::2] + i
                    bboxes = np.concatenate([bboxes, new_bboxes], axis=0)
        return bboxes

    @staticmethod
    def plot_predictions(image, bboxes):
        import pylab as plt
        plt.imshow(image)
        current_axis = plt.gca()
        for bbox in bboxes:
            coords = (bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
            current_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
        plt.show()

def is_white(img, max_diff_between_chanells = 2, min_channel_value = 245.0):
    max_channel_var = np.var([-max_diff_between_chanells, max_diff_between_chanells, 0])
    return np.bitwise_and(img.var(axis=2) < max_channel_var, img.min(axis = 2) < min_channel_value)

def fill_detections(self, img, bboxes):
    mask = is_white(img)
    


if __name__ == "__main__":
    image = caffe.io.load_image('000.jpg')
    detector = TextDetector()
    bboxes = detector.get_predictions_full_image(image)

    TextDetector.plot_predictions(image, bboxes)
