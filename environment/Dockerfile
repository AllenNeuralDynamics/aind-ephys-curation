# hash:sha256:b0370d70bcd053e8f98ad98c3978d930db21d2c6d64907111c1f770a1d982737
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.17.1 \
    spikeinterface[full]==0.99.1
