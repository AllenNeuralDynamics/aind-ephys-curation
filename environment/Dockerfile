# hash:sha256:5fd9fbf0d7986bd30fa93c0a3578b529937b20cda8272d160d8537a957dfa716
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.3 \
    spikeinterface[full]==0.101.2
