# hash:sha256:65e2d1cb33c76317af394b37473292d96cd13a3134de72b4d0bc9621d08dd31c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab==4.3.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    scikit-learn==1.4.2 \
    spikeinterface[full]==0.103.2
