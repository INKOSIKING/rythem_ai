$ErrorActionPreference = "Stop"

$ORG_NAME = "rythemai"
$REPOS = @("music-ai-models", "music-ai-backend", "music-ai-sdk-js", "music-ai-web", "music-ai-mobile", "music-ai-devportal")

foreach ($repo in $REPOS) {
    Write-Host "==> Creating repo folder: $repo"
    New-Item -ItemType Directory -Force -Path $repo | Out-Null
    Set-Location $repo

    git init -b main
    "# $repo" | Out-File -Encoding utf8 README.md

    Invoke-WebRequest "https://www.toptal.com/developers/gitignore/api/python,node,react,visualstudiocode,linux,macos,windows" -OutFile .gitignore
    Invoke-WebRequest "https://raw.githubusercontent.com/github/choosealicense.com/gh-pages/_licenses/mit.txt" -OutFile LICENSE

    switch ($repo) {
        "music-ai-models" {
            New-Item -ItemType Directory -Name "models","scripts","data","tests" | Out-Null
            New-Item -ItemType File -Name "requirements.txt" | Out-Null
        }
        "music-ai-backend" {
            New-Item -ItemType Directory -Name "app","tests","configs" | Out-Null
            New-Item -ItemType File -Name "requirements.txt","Dockerfile" | Out-Null
        }
        "music-ai-sdk-js" {
            New-Item -ItemType Directory -Name "src","tests" | Out-Null
            npm init -y | Out-Null
        }
        "music-ai-web" {
            npx create-react-app . --template typescript
            npm install --save music-ai-sdk
        }
        "music-ai-mobile" {
            npx react-native init MusicAIMobile
            npm install --save music-ai-sdk
        }
        "music-ai-devportal" {
            npx create-next-app@latest . --typescript
            New-Item -ItemType Directory -Name "docs","examples" | Out-Null
        }
    }
    Set-Location ..
}
Write-Host "==> All repo scaffolds created successfully."