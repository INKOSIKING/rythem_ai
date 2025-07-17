$ErrorActionPreference = "Stop"
Write-Host "==> Running linter and tests..."
npm run lint
npm test

Write-Host "==> Building package..."
npm run build

Write-Host "==> Publishing to npm..."
npm publish --access public

Write-Host "==> Done."