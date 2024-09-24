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


embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.long),
}
torch.onnx.export(
    Decoder(decoder),
    tuple(dummy_inputs.values()),
    decoder_path,
    input_names=list(dummy_inputs.keys()),
    output_names=["masks"],
    opset_version=16,
    do_constant_folding=True,
)
os.system("pip install onnx onnx_graphsurgeon polygraphy")
os.system(f"polygraphy surgeon sanitize --fold-constants {decoder_path} -o {decoder_path}")


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
