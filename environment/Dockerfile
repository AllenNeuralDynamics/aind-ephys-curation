# hash:sha256:d3b207bc17c257a2badf3a5c82237e18f58e7067a02d64e1031e6b4612e25706
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.13.50

COPY postInstall /
RUN /postInstall
