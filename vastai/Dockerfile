FROM vastai/base-image:cuda-12.9.1-cudnn-devel-ubuntu24.04-py311-2026-03-26

ENV PATH="/venv/main/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT="/venv/main"

# Clone repo
WORKDIR /opt/workspace-internal
RUN git clone https://github.com/amk16/openpi-Vega3D

# Install dependencies
WORKDIR /opt/workspace-internal/openpi-Vega3D
RUN --mount=type=cache,target=/root/.cache/uv GIT_LFS_SKIP_SMUDGE=1 uv sync
RUN --mount=type=cache,target=/root/.cache/uv GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Custom aliases
RUN echo "alias nvidia-smi='watch -n 1 nvidia-smi'" >> /root/.bashrc && \
    echo "alias du='du --max-depth=1 -h'" >> /root/.bashrc

# Onstart script — point Vast.ai template to: bash /opt/scripts/onstart.sh
COPY onstart.sh /opt/scripts/onstart.sh
RUN chmod +x /opt/scripts/onstart.sh
