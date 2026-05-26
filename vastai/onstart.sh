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
