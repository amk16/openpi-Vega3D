# Dockerfile for the LIBERO benchmark.

# Build the container:
# docker build . -t libero -f examples/libero/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all libero /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    git \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Clone LIBERO-Pro (replaces third_party/libero)
RUN git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git /opt/libero-pro

# Copy the requirements files so we can install dependencies.
COPY ./examples/libero/requirements.txt /tmp/requirements.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN uv venv --python 3.8 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
RUN uv pip install -e /opt/libero-pro
ENV PYTHONPATH=/app:/app/packages/openpi-client/src

# LIBERO-Pro config
ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /opt/libero-pro/libero/libero
bddl_files: /opt/libero-pro/libero/libero/bddl_files
init_states: /opt/libero-pro/libero/libero/init_files
datasets: /opt/libero-pro/libero/datasets
assets: /opt/libero-pro/libero/libero/assets
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/libero/main.py $CLIENT_ARGS"]
