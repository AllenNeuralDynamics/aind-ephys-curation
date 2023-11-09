# hash:sha256:257eae52237526fa847b7486a36872f5abb6c4e88a0184635225f1d6cb6976fe
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25

COPY postInstall /
RUN /postInstall
