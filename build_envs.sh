#Build virtual environment
python -m venv ./.venv
source .venv/bin/activate


# Install required pip packages
cd DSC180A_JEDI
pip install -r requirements.txt
python -m pip install pudb
python -m pip install datasets
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install git+https://github.com/huggingface/transformers
python -m pip install nltk==3.7 
python -m pip install boto3 sacremoses tensorboardX torchdiffeq einops npy_append_array 
python -m pip install -i https://testpypi.python.org/pypi peft
python -m pip install -U pip setuptools

cd ..
# Install Apex
rm -rf apex
git clone https://github.com/NVIDIA/apex
cp DSC180A_JEDI/apex_utils/setup.py apex
cd apex

python setup.py install --cuda_ext #This will take a while

cd ..
echo "Complete"

