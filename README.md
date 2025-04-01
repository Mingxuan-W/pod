# POD : Predict-Optimize-Distill
Predict-Optimize-Distill: A Self-Improving Cycle for 4D Object Understanding
![image](assets/teaser.jpg)

# Environment Setup
```
git clone --recursive https://github.com/Mingxuan-W/pod.git
```

## create conda env
``` 
# create pod env and nerfstudio related environment
conda create --name pod -y python=3.10 
conda activate pod
- pip install markdown packaging protobuf six werkzeug pandas
- pip install numpy==1.26.4
- pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
- pip install nerfstudio
- conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
- pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## build garfiled 
```
- cd dependencies/garfield
- pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.2.2 cuml-cu12==24.2.0  cupy==13.3.0
- pip install -e.
```

## build dig
```
- cd dependencies/dig
- pip install -e.
```

## build pod
```
- pip install -e.
```

# Running POD

## 1. RSRD Init
``` 
- python scripts/run_rsrd_init.py  --video_idx 1 --dataset_name toy
``` 
## 2. Pod Pipeline
``` 
- python scripts/run_pod_pipeline.py --video_idx 0 --dataset_name toy2 --preload_camera --run_name default --rsrd_path video/toy2_long/toy2_2/init/rsrd_/init_pose.pt
``` 