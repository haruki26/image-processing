FROM ghcr.io/astral-sh/uv:0.11.7-python3.14-trixie

RUN apt-get update && apt-get install -y \
    curl sudo git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=${USER_ID}
ARG USER_NAME=vscode

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -d /home/vscode -s /bin/bash -u ${USER_ID} -g ${USER_NAME} ${USER_NAME} && \
    echo ${USER_NAME}' ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV HOME=/home/${USER_NAME}
USER ${USER_NAME}
