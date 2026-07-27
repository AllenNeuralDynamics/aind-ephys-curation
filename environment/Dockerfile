# hash:sha256:f88c884f7be3c09e5c1487b7066d9dcbab145c9db71d8be7bc4ea476a5050264
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

RUN --mount=type=secret,id=secrets . /run/secrets/secrets
