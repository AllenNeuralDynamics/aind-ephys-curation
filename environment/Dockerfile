# hash:sha256:cbf3da5a6af6216f98bece9253830b68faf4cd60a120a5556aae50f233cb49a3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.3.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.103.0
