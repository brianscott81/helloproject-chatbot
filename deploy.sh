#!/usr/bin/env bash
# Deploy the Hello! Project Wiki chatbot to Fly.io.
#
# Prerequisites:
#   1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
#   2. Sign in: fly auth signup   (or: fly auth login)
#   3. Have helloproject.db and chroma/ built locally.
#
# Usage:
#   ./deploy.sh                    # full deploy
#   ./deploy.sh --no-llm           # deploy in template-only mode
#   ./deploy.sh --logs             # tail logs after deploy
#
# This script is idempotent — running it twice is safe.

set -euo pipefail

# Parse flags
DEPLOY_FLAGS=()
TAIL_LOGS=false
for arg in "$@"; do
    case "$arg" in
        --no-llm)    DEPLOY_FLAGS+=(--no-llm) ;;
        --logs)      TAIL_LOGS=true ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# Verify required artifacts exist
if [ ! -f helloproject.db ]; then
    echo "Error: helloproject.db not found." >&2
    echo "Build it first: python build_index.py" >&2
    exit 1
fi
if [ ! -d chroma ]; then
    echo "Error: chroma/ directory not found." >&2
    echo "Build it first: python build_embeddings.py" >&2
    exit 1
fi

# Detect flyctl
if ! command -v fly >/dev/null 2>&1; then
    echo "Error: flyctl not found. Install from https://fly.io/docs/hands-on/install-flyctl/" >&2
    exit 1
fi

# Check if this is the first deploy
if ! fly status >/dev/null 2>&1; then
    echo "App not deployed yet. Running 'fly launch' (no-deploy)..."
    fly launch --no-deploy
    echo ""
    echo "If fly launch asked for an app name, edit 'app = ...' in fly.toml."
    echo "If you declined to create a database, you're good. Otherwise, run:"
    echo "  fly volumes create helloproject_data --size 1"
    echo ""
    read -p "Continue with deploy? [y/N] " resp
    if [ "$resp" != "y" ] && [ "$resp" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Deploy
echo "==> Deploying to Fly.io..."
fly deploy "${DEPLOY_FLAGS[@]}"

# Show the URL
echo ""
echo "==> Done! Your app is at:"
fly status | grep -E "^Hostname|^URL" || true
fly info 2>/dev/null | grep -E "Hostname" || true

if [ "$TAIL_LOGS" = true ]; then
    echo ""
    echo "==> Tailing logs (Ctrl-C to stop)..."
    fly logs
fi
