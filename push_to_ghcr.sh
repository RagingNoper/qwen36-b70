#!/usr/bin/env bash
# One-time: push the local image to GHCR so others can `docker pull` it.
# Needs a token with write:packages — step 1 grants it (opens a browser to approve).
set -e
gh auth refresh -h github.com -s write:packages          # 1. approve in browser
gh auth token | docker login ghcr.io -u RagingNoper --password-stdin   # 2. log in
docker tag qwen36-b70-ship:latest ghcr.io/ragingnoper/qwen36-b70-ship:latest  # 3. tag
docker push ghcr.io/ragingnoper/qwen36-b70-ship:latest   # 4. push (~11 GB upload)
echo
echo "Now make it PUBLIC: github.com/users/RagingNoper/packages/container/qwen36-b70-ship/settings"
echo "  -> Danger Zone -> Change visibility -> Public"
