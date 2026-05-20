# syntax=docker/dockerfile:1
FROM vastai/base-image:cuda-12.9.1-cudnn-devel-ubuntu24.04-py311-2026-03-26

ENV PATH="/venv/main/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT="/venv/main"

# Copy lockfiles to bust cache when deps change
COPY pyproject.toml uv.lock /tmp/.cache-trigger/

# Clone repo
WORKDIR /opt/workspace-internal
RUN git clone https://github.com/amk16/openpi-Vega3D

# Install dependencies
WORKDIR /opt/workspace-internal/openpi-Vega3D
RUN --mount=type=cache,target=/root/.cache/uv GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN --mount=type=cache,target=/root/.cache/uv GIT_LFS_SKIP_SMUDGE=1 uv pip install --system -e .
WORKDIR /opt/workspace-internal

# Custom aliases
RUN echo "alias nvidia-smi='watch -n 1 nvidia-smi'" >> /root/.bashrc && \
    echo "alias du='du --max-depth=1 -h'" >> /root/.bashrc

# Onstart script — point Vast.ai template to: bash /opt/scripts/onstart.sh
COPY vastai/onstart.sh /opt/scripts/onstart.sh
RUN chmod +x /opt/scripts/onstart.sh

RUN touch ~/.no_auto_tmux

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip

# Install Node.js + npm + Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs
RUN npm install -g @anthropic-ai/claude-code

RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-cli

# === LIBERO eval environment (Python 3.8, separate from main openpi env) ===

# System deps for MuJoCo + OpenGL rendering
RUN apt-get update && apt-get install -y \
    libosmesa6-dev libgl1-mesa-glx libegl1 libglew-dev \
    libglfw3-dev libgles2-mesa-dev libglib2.0-0 \
    libsm6 libxrender1 libxext6 make g++ clang

# Python 3.8 venv for LIBERO (uv venv omits pip, so bootstrap it first)
RUN uv venv --python 3.8 /venv/libero
RUN /venv/libero/bin/python -m ensurepip --upgrade

# Clone LIBERO-Pro
RUN git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git /opt/libero-pro

# Install LIBERO-Pro deps + package
COPY examples/libero/requirements.txt /tmp/requirements-libero.txt
RUN /venv/libero/bin/pip3 install -r /tmp/requirements-libero.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113
RUN /venv/libero/bin/pip3 install -e /opt/libero-pro

# Install openpi-client in libero venv (for websocket connection)
RUN /venv/libero/bin/pip3 install /opt/workspace-internal/openpi-Vega3D/packages/openpi-client

# MuJoCo + EGL config
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# LIBERO config
ENV LIBERO_CONFIG_PATH=/opt/libero-config
RUN mkdir -p /opt/libero-config && cat <<'EOF' > /opt/libero-config/config.yaml
benchmark_root: /opt/libero-pro/libero/libero
bddl_files: /opt/libero-pro/libero/libero/bddl_files
init_states: /opt/libero-pro/libero/libero/init_files
datasets: /opt/libero-pro/libero/datasets
assets: /opt/libero-pro/libero/libero/assets
EOF
