#!/bin/bash
set -euo pipefail

REPO_NAME=$(basename $(pwd))
cat <<EOF > README.md
# $REPO_NAME

$(head -n 1 LICENSE)

## Setup

\`\`\`bash
# Install dependencies
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python main.py
\`\`\`
EOF
echo "README.md generated."