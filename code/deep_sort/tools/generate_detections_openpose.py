# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
import mmcv
import json
import math

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out_dad = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out_dad, batch_size)

        ##### ADD NEW FEATURES HERE ####

        out_ = np.zeros((len(data_x), 22), np.float32)
        for i in range(len(data_x)):
            out_[i,:] = 0

        out = np.concatenate((out_dad,out_), axis=1)


        out = out_dad

        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, offset, openpose=''):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        if not sequence.startswith('.'):
            print("Processing %s" % sequence)
            sequence_dir = os.path.join(mot_dir, sequence)

            image_dir = os.path.join(sequence_dir, "img1")
            image_filenames = {
                int(os.path.splitext(f.replace("frame",""))[0])-offset: os.path.join(image_dir, f)
                for f in os.listdir(image_dir)}

            detections_out = []

            n_str_frame = 0
            max_frame_idx = len(image_filenames) - 1
            for frame_idx in range(0, max_frame_idx + 1):
                aaa = int((max_frame_idx + 1)/ 100)
                if frame_idx % aaa == 0:
                    print("Processing frame {} / {}".format(frame_idx, max_frame_idx))

                if frame_idx not in image_filenames:
                    print("WARNING could not find image for frame %d" % frame_idx)
                    continue
                bgr_image = cv2.imread(
                    image_filenames[frame_idx], cv2.IMREAD_COLOR)

                width = bgr_image.shape[1]
                height = bgr_image.shape[0]
                #####

                def str_frame(n):
                    str_frame = ""
                    for _ in range(n): str_frame += "0"
                    return str_frame


                rows = []
                ### OPEN POSE DATA
                if frame_idx == 0:
                    while not os.path.exists(os.path.join(sequence_dir,openpose,sequence+"_" + str_frame(n_str_frame) + str(frame_idx+offset)+"_keypoints.json")) and n_str_frame<30:
                        n_str_frame += 1
                    if n_str_frame==30:
                        print("Could not find openpose data")
                        exit()

                def log(n):
                    if n == 0:
                        return 0
                    else:
                        return math.floor(math.log(n, 10))

                with open(os.path.join(sequence_dir, openpose, sequence + "_" + str_frame(
                        n_str_frame - log(frame_idx + offset) + log(offset)) + str(
                        frame_idx + offset) + "_keypoints.json")) as json_file:
                    openpose_data = json.load(json_file)
                    #print("Players :", len(openpose_data['people']))
                    for p in openpose_data['people']:
                        keypoints = np.array(p["pose_keypoints_2d"]).reshape((-1, 3))
                        keypoints = np.array(
                            [kp for kp in list(keypoints) if kp[2] > 0.1 and not (kp[0] == -1.0 and kp[1] == -1.0)])
                        if keypoints.shape[0] >= 1:
                            #print("Keypoints: ", keypoints.shape[0])
                            bbox_center = [
                                  (np.mean(keypoints[:, 0]) + 1) / 2 * width,
                                  (np.mean(keypoints[:, 0]) + 1) / 2 * height]

                            bbox = [
                                  (np.min(keypoints[:, 0]) + 1) / 2 * width,
                                  (np.min(keypoints[:, 1]) + 1) / 2 * height,
                                  (np.max(keypoints[:, 0]) + 1) / 2 * width,
                                  (np.max(keypoints[:, 1]) + 1) / 2 * height]
                            bbox[2] -= bbox[0]
                            bbox[3] -= bbox[1]
                            rows.append([frame_idx, -1] + bbox + [np.mean(keypoints[:, 2]), -1, -1, -1])

                #####
                rows = np.array(rows)
                features = encoder(bgr_image, rows[:, 2:6].copy())
                detections_out += [np.r_[(row, feature)] for row, feature
                                   in zip(rows, features)]

            output_filename = os.path.join(output_dir, "{}_{}.npy".format(sequence, "detbyop"))
            np.save(
                output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--offset", help="Frame offset. Default to 0", default=0)
    parser.add_argument(
        "--openpose", help="RELATIVE path to openpose features. Default to empty string (openpose features are not used in this case)", default='')
    parser.add_argument(
        "--det_stage", help="Detection stage id", default='detbyop')
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir, int(args.offset), args.openpose)


if __name__ == "__main__":
    main()
