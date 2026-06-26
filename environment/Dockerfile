# hash:sha256:e5772ade4c85b7108d48f425f65ef0ae769c210f5d0bb416e3203cf39748633f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab==4.3.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-log-utils==0.2.3 \
    scikit-learn==1.4.2 \
    spikeinterface[full]==0.104.7
