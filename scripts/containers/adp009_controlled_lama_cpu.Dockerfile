FROM python:3.10-slim@sha256:34a2c9467a0231d8c29a5ecadc219733a9393b026882b44d91616b9dae6088b6
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir \
    torch==1.13.1 torchvision==0.14.1 \
    numpy==1.23.5 pyyaml==6.0.2 tqdm==4.67.1 easydict==1.9.0 \
    scikit-image==0.19.3 scikit-learn==1.1.3 scipy==1.10.1 \
    opencv-python-headless==4.7.0.72 joblib==1.2.0 matplotlib==3.7.5 \
    pandas==1.5.3 albumentations==0.5.2 hydra-core==1.1.2 \
    pytorch-lightning==1.2.9 tabulate==0.9.0 kornia==0.5.0 \
    webdataset==0.1.103 packaging==24.2 tensorboard==2.11.2
WORKDIR /source
