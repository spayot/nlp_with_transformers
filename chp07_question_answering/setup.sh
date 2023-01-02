conda create -n tfm7 python==3.10.0
conda activate tfm7
pip install -r requirements.txt
GRPC_PYTHON_BUILD_SYSTEM_ZLIB=true pip install git+https://github.com/deepset-ai/haystack.git