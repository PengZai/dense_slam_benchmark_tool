# Just Read Me



# install prior-depth-anything envrionment
pip install torch==2.2.2 torchvision==0.17.2
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install -e ".[prior-depth-anything]"



# install vggt-long
python -m pip install -e . --dry-run --no-build-isolation
python -m py_compile setup.py vggt_long.py

# install benchmark dataset
conda create -n benchmark-dataset python=3.11.0
conda activate benchmark-dataset
python3 -m pip install -e .
python3 -m pip install thridparity/in3d