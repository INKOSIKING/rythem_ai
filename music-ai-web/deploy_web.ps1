$ErrorActionPreference = "Stop"
Write-Host "==> Building web app..."
npm run build

Write-Host "==> Deploying to Vercel..."
vercel --prod

Write-Host "==> Done."