# hash:sha256:cf6777be473b4c2255a6201391cb74c336ef37d92f761aa747d565d255a734c1
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
    log-schema==0.2.10.dev1 \
    scikit-learn==1.4.2 \
    spikeinterface[full]==0.104.3
