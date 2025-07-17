#!/bin/bash
set -euo pipefail

echo "==> Building web app..."
npm run build

echo "==> Deploying to Vercel..."
vercel --prod

echo "==> Done."