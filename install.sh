source conda activate
conda create -n ML python=3.10.8
conda activate ML

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt