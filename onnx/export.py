import os
import argparse
from pathlib import Path

import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel


argparser = argparse.ArgumentParser()
argparser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory")
args = argparser.parse_args()

ROOT_DIR = Path(__file__).resolve().parents[1].as_posix()
checkpoint = f"{ROOT_DIR}/weights/mobile_sam.pt"
model_type = "vit_t"
encoder_path = f"{args.output_dir}/encoder.onnx"
decoder_path = f"{args.output_dir}/decoder.onnx"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
encoder = sam.image_encoder
decoder = SamOnnxModel(sam, return_single_mask=True)
encoder.eval()
decoder.eval()

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}
embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.long),
}
torch.onnx.export(
    decoder,
    tuple(dummy_inputs.values()),
    decoder_path,
    input_names=list(dummy_inputs.keys()),
    output_names=["masks", "scores", "logits"],
    opset_version=16,
    do_constant_folding=True,
    dynamic_axes=dynamic_axes,
)
# os.system("pip install onnx onnx_graphsurgeon polygraphy")
# os.system(f"polygraphy surgeon sanitize --fold-constants {decoder_path} -o {decoder_path}")


#### Encoder
im = torch.randn(1, 3, sam.image_encoder.img_size, sam.image_encoder.img_size)
torch.onnx.export(
    encoder,
    im,
    encoder_path,
    input_names=["image"],
    output_names=["image_embeddings"],
    opset_version=16,
    do_constant_folding=True,
)
# os.system(f"polygraphy surgeon sanitize --fold-constants {encoder_path} -o {encoder_path}")
