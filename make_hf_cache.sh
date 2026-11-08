# This script creates the directory structure for Hugging Face's cache.
# Ignore this script if you are ok with HF defaults.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
HF_DIR="$SCRIPT_DIR/hf_cache"
BASHRC="$HOME/.bashrc"

mkdir -p "$HF_DIR"
echo "Created cache folder: $HF_DIR"

EXPORT_LINE="export HF_HOME=\"$HF_DIR\""

touch "$BASHRC"

if grep -q '^export HF_HOME=' "$BASHRC"; then
    sed -i.bak "s|^export HF_HOME=.*|$EXPORT_LINE|" "$BASHRC"
    echo "Updated existing HF_HOME in $BASHRC (backup at $BASHRC.bak)"
else
    printf '\n# Hugging Face cache location\n%s\n' "$EXPORT_LINE" >> "$BASHRC"
    echo "Added HF_HOME to $BASHRC"
fi

echo "Run 'source ~/.bashrc' or open a new shell to apply it."
