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


class Decoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, image_embeddings, point_coords, orig_im_size) -> torch.Tensor:
        point_labels = torch.tensor([[2, 3]], dtype=torch.float)
        mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float)
        has_mask_input = torch.zeros([1], dtype=torch.float)
        masks, scores, logits = self.decoder(
            image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_im_size
        )
        return masks


class Wrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image, point_coords, orig_im_size) -> torch.Tensor:
        image_embeddings = self.encoder(image)
        point_labels = torch.tensor([[2, 3]], dtype=torch.float)
        mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float)
        has_mask_input = torch.zeros([1], dtype=torch.float)
        masks, scores, logits = self.decoder(
            image_embeddings, point_coords, point_labels, mask_input, has_mask_input, orig_im_size
        )
        return masks


image = torch.randn(1, 3, sam.image_encoder.img_size, sam.image_encoder.img_size)
point_coords = torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float)
orig_im_size = torch.tensor([1500, 2250], dtype=torch.long)

model = Wrapper(encoder, decoder)
model.eval()

model_path = f"{args.output_dir}/sam.onnx"
torch.onnx.export(
    model,
    (image, point_coords, orig_im_size),
    model_path,
    input_names=["image", "point_coords", "orig_im_size"],
    output_names=["masks"],
    opset_version=16,
    do_constant_folding=True,
)
os.system("pip install onnx onnx_graphsurgeon polygraphy")
os.system(f"polygraphy surgeon sanitize --fold-constants {model_path} -o {model_path}")
