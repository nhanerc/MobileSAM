import os
import typing as T
import argparse
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]

argparser = argparse.ArgumentParser()
argparser.add_argument("--encoder", "-e", type=str, default="", help="path to the encoder model")
argparser.add_argument("--decoder", "-d", type=str, default="", help="path to the decoder model")
argparser.add_argument("--output_dir", "-o", type=str, required=True, help="output directory")

args = argparser.parse_args()
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


def run(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    image_path: str,
    output_path: str,
    point_coords: T.Optional[np.ndarray] = None,
    point_labels: T.Optional[np.ndarray] = None,
    box: T.Optional[np.ndarray] = None,
    mask_input: T.Optional[np.ndarray] = None,
) -> None:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    viz_image = image.copy()
    orig_im_size = image.shape[:2]

    resizer = ResizeLongestSize(MAX_SIZE)
    image = resizer.apply_image(image)
    image = preprocess(image, MAX_SIZE)

    onnx_coords = []
    onnx_labels = []
    if point_coords is not None:
        assert len(point_coords) == len(point_labels)
        onnx_coords.extend(point_coords)
        onnx_labels.extend(point_labels)

    if box is None:
        # padding point
        onnx_coords.append([0, 0])
        onnx_labels.append(-1)
    else:
        assert len(box) == 4
        onnx_coords.extend(box.reshape(2, 2))
        onnx_labels.extend([2, 3])

    if mask_input is None:
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    else:
        assert mask_input.shape == (1, 1, 256, 256)
        onnx_mask_input = mask_input.astype(np.float32)
        onnx_has_mask_input = np.ones(1, dtype=np.float32)

    onnx_coords = np.array(onnx_coords, dtype=np.float32)[None, :, :]
    onnx_labels = np.array(onnx_labels, dtype=np.float32)[None, :]

    onnx_coords = resizer.apply_coords(onnx_coords, orig_im_size)

    # Encode the image
    image_embeddings = encoder.run(None, {encoder.get_inputs()[0].name: image})[0]

    # Decode the image
    masks, scores, logits = decoder.run(
        None,
        {
            "image_embeddings": image_embeddings,
            "point_coords": onnx_coords,
            "point_labels": onnx_labels,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(orig_im_size, dtype=np.int64),
        },
    )

    print(output_path)
    print(masks.shape, scores.shape, logits.shape)
    print(scores)
    print()
    masks = (masks > 0.0).astype(np.uint8).squeeze()

    # addWeighted expects 3 channels
    masks = np.stack([masks] * 3, axis=-1) * np.random.randint(0, 255, size=(1, 1, 3), dtype=np.uint8)
    alpha = 0.8
    viz_image = cv2.addWeighted(viz_image, alpha, masks, 1 - alpha, 0)

    # # visualize the mask
    # contours = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # contour = max(contours, key=cv2.contourArea)
    # cv2.polylines(viz_image, [contour], 1, (0, 255, 0), 1)

    if box is not None:
        x1, y1, x2, y2 = box.astype(np.int32)
        cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # blue

    if point_coords is not None:
        for point, label in zip(point_coords, point_labels):
            x, y = point.astype(np.int32)
            if label == 1:
                color = (0, 255, 0)  # green
            else:
                color = (255, 0, 0)  # red
            cv2.circle(viz_image, (x, y), 5, color, -1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    args = argparser.parse_args()
    encoder = ort.InferenceSession(args.encoder)
    decoder = ort.InferenceSession(args.decoder)

    image_path = f"{ROOT_DIR}/notebooks/images/picture2.jpg"
    box = np.array([210, 200, 350, 500], dtype=np.float32)
    point_coords = np.array([[250, 375], [375, 360]])
    point_labels = np.array([1, 0])

    output_path = f"{args.output_dir}/box.jpg"
    run(encoder, decoder, image_path, output_path, box=box)

    output_path = f"{args.output_dir}/points.jpg"
    run(encoder, decoder, image_path, output_path, point_coords=point_coords, point_labels=point_labels)

    output_path = f"{args.output_dir}/box_points.jpg"
    run(
        encoder,
        decoder,
        image_path,
        output_path,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
    )
