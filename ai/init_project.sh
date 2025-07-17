#!/bin/bash
set -euo pipefail

ORG_NAME="rythemai"
REPOS=("music-ai-models" "music-ai-backend" "music-ai-sdk-js" "music-ai-web" "music-ai-mobile" "music-ai-devportal")

echo "==> Creating repo folders..."
for repo in "${REPOS[@]}"; do
  mkdir -p "$repo"
  cd "$repo"

  echo "==> Initializing $repo"
  git init -b main
  echo "# $repo" > README.md

  # Add standard .gitignore and LICENSE
  curl -fsSL https://www.toptal.com/developers/gitignore/api/python,node,react,visualstudiocode,linux,macos,windows > .gitignore
  curl -fsSL https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt > LICENSE

  # Add repo-specific folders
  case $repo in
    "music-ai-models")
      mkdir -p models scripts data tests
      touch requirements.txt
      ;;
    "music-ai-backend")
      mkdir -p app tests configs
      touch requirements.txt Dockerfile
      ;;
    "music-ai-sdk-js")
      mkdir -p src tests
      npm init -y
      ;;
    "music-ai-web")
      npx create-react-app . --template typescript
      npm install --save music-ai-sdk
      ;;
    "music-ai-mobile")
      npx react-native init MusicAIMobile
      npm install --save music-ai-sdk
      ;;
    "music-ai-devportal")
      npx create-next-app@latest . --typescript
      mkdir -p docs examples
      ;;
  esac
  cd ..
done

echo "==> All repo scaffolds created successfully."