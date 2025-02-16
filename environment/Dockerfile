# hash:sha256:de132c0401f4c96a1fabad1a40268cf9db504e376cd4316076a959953a4d4fe9
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl=7.68.0-1ubuntu2.19 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN /postInstall
