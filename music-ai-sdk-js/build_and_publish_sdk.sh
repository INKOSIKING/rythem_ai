#!/bin/bash
set -euo pipefail

echo "==> Running linter and tests..."
npm run lint
npm test

echo "==> Building package..."
npm run build

echo "==> Publishing to npm..."
npm publish --access public

echo "==> Done."