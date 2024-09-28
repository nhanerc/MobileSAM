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
python onnx/export.py -o weights

# 4. Run inference with ONNX
python onnx/inference.py -e weights/encoder.onnx -d weights/decoder.onnx -o result
```