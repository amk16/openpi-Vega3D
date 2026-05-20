#!/bin/bash

# Git credentials (set GITHUB_TOKEN as env var in Vast.ai template)
git config --global credential.helper store
echo "https://oauth2:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
git config --global user.name "Salman Shahid"
git config --global user.email "salmanshahidzia@gmail.com"

# Only copy if not already there (persists across restarts)
if [ ! -d /workspace/openpi-Vega3D ]; then
    cp -r /opt/workspace-internal/openpi-Vega3D /workspace/openpi-Vega3D
fi

cd /workspace/openpi-Vega3D

# Download LIBERO-Pro BDDL/init files (one-time, persists across restarts)
if [ ! -f /opt/libero-pro/.data_downloaded ]; then
    echo "Downloading LIBERO-Pro data files..."
    /venv/libero/bin/pip install huggingface_hub
    /venv/libero/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('zhouxueyang/LIBERO-Pro', local_dir='/tmp/libero-pro-data', repo_type='dataset')
"
    cp -r /tmp/libero-pro-data/bddl_files/* /opt/libero-pro/libero/libero/bddl_files/
    cp -r /tmp/libero-pro-data/init_files/* /opt/libero-pro/libero/libero/init_files/
    touch /opt/libero-pro/.data_downloaded
    rm -rf /tmp/libero-pro-data
    echo "LIBERO-Pro data downloaded successfully."
fi
