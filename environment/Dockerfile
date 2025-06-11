# hash:sha256:8fc38ce2c12d9c8f2ccf29c9edacba68099589289f314e74f1a02aa5d14a4107
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.3.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.102.3
