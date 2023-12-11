#Build virtual environment
cd ~  #cd to workspace/path
python -m .venv "path/to/new/virtual/environment"
source .venv/bin/activate




# Install required pip packages
cd DSC180A_JEDI
pip install -r requirements.txt
python -m pip install git+https://github.com/huggingface/transformers
python -m pip install -i https://testpypi.python.org/pypi peft
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U pip setuptools

cd ..
# Install Apex
rm -rf apex
git clone https://github.com/NVIDIA/apex
mv DSC180A_JEDI/apex_utils/setup.py apex
cd apex
#pip install -v --disable-pip-version-check --no-cache-dir ./
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
python setup.py install --cuda_ext #This will take a while

cd ../DSC180A_JEDI
echo "Complete"

