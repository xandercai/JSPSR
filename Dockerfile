FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y openssh-server tzdata python3-dev gdal-bin libgdal-dev zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Pacific/Auckland \
    SHELL=bash

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip uninstall numba tensorboard -y && \
    python -m pip install --no-cache-dir numba tensorboard torchinfo timm natsort easydict scikit-image seaborn rasterio rioxarray geopandas richdem mapply piq kornia hide-warnings black imagecodecs pandas==1.5.*

ENTRYPOINT ["tail", "-f", "/dev/null"]


