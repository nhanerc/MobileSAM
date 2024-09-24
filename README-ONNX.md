```bash
# 1. Clone the project
git clone https://github.com/nhanerc/MobileSAM.git
cd MobileSAM
git checkout onnx

# 2. Use docker container
docker run -it --rm --gpus all -v `pwd`:/workspace -w /workspace nvcr.io/nvidia/pytorch:24.07-py3
pip install -e .
pip install onnxruntime timm

# 3. Export to ONNX
# python onnx/export_mixed.py -o weights # accept points or/and boxes
python onnx/export_box_separate.py -o weights # accept a box per loop
python onnx/export_box_merged.py -o weights # merged encoder and decoder, accept a box per loop

# 4. Run inference with ONNX
python onnx/inference.py -e weights/encoder.onnx -d weights/decoder.onnx -o result
python onnx/inference.py -m weights/sam.onnx -o result
```