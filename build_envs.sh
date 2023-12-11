
# Install required pip packages
# cd /data/jieqi/DSC180A_JEDI
python -m pip install git+https://github.com/huggingface/transformers
python -m pip install nltk==3.7 
python -m pip install boto3 sacremoses tensorboardX torchdiffeq einops npy_append_array 
python -m pip install pudb accelerate datasets
python -m pip install -i https://testpypi.python.org/pypi peft
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U pip setuptools

cd ..
# Install Apex
rm -rf apex
git clone https://github.com/NVIDIA/apex
cd apex
#pip install -v --disable-pip-version-check --no-cache-dir ./
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
python setup.py install --cuda_ext
cd ..
