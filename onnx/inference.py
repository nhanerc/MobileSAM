import os
import argparse
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", "-m", type=str, default="", help="path to the merged model")
argparser.add_argument("--encoder", "-e", type=str, default="", help="path to the encoder model")
argparser.add_argument("--decoder", "-d", type=str, default="", help="path to the decoder model")
argparser.add_argument("--output_dir", "-o", type=str, required=True, help="output directory")

args = argparser.parse_args()
merged = args.model and os.path.exists(args.model)
separate = args.encoder and args.decoder and os.path.exists(args.encoder) and os.path.exists(args.decoder)
assert merged != separate, "Either provide a merged model or both encoder and decoder models"

MAX_SIZE = 1024


class ResizeLongestSize:
    """
    Copied from segment_anything and modify based on available dependencies.

    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        h, w = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return cv2.resize(image, (w, h))

    def apply_coords(self, coords: np.ndarray, original_size) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww


def preprocess(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    nh, nw = image.shape[:2]
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.astype(np.float32) / 255
    image = (image - mean) / std

    # pad to (max_size, max_size, 3)
    pad_h = max_size - nh
    pad_w = max_size - nw
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)

    # HWC to CHW
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    return image


image_path = f"{ROOT_DIR}/notebooks/images/picture2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
viz_image = image.copy()
orig_im_size = image.shape[:2]

resizer = ResizeLongestSize(MAX_SIZE)
image = resizer.apply_image(image)
image = preprocess(image, MAX_SIZE)

input_box = np.array([210, 200, 350, 500], dtype=np.float32)
point_coords = resizer.apply_coords(input_box.reshape(1, 2, 2), orig_im_size)

if separate:
    encoder = ort.InferenceSession(args.encoder)
    decoder = ort.InferenceSession(args.decoder)

    # Encode the image
    image_embeddings = encoder.run(None, {encoder.get_inputs()[0].name: image})[0]

    # Decode the image
    masks = decoder.run(
        None,
        {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "orig_im_size": np.array(orig_im_size, dtype=np.int64),
        },
    )[0]
    output_path = f"{args.output_dir}/separate.jpg"
else:
    model = ort.InferenceSession(args.model)
    masks = model.run(
        None,
        {
            "image": image,
            "point_coords": point_coords,
            "orig_im_size": np.array(orig_im_size, dtype=np.int64),
        },
    )[0]
    output_path = f"{args.output_dir}/merged.jpg"

masks = (masks > 0.5).astype(np.uint8)
masks = masks.squeeze()
contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x1, y1, x2, y2 = input_box.astype(np.int32)
cv2.rectangle(viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.polylines(viz_image, contours, True, (0, 255, 0), 1)

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
cv2.imwrite(output_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
