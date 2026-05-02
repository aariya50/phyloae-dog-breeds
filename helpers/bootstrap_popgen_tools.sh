#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/.tools"
BIN_DIR="${TOOLS_DIR}/bin"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cbmf4761-popgen.XXXXXX")"

cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

mkdir -p "${BIN_DIR}"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detected: OS=${OS}, ARCH=${ARCH}"

# ---------- plink2 ----------
echo "[1/2] Installing plink2..."

case "${OS}-${ARCH}" in
  Darwin-arm64)
    PLINK_URL="$(curl -fsSL https://www.cog-genomics.org/plink/2.0/ \
      | sed -n 's/.*href="\(https:\/\/s3.amazonaws.com\/plink2-assets\/plink2_mac_arm64_[0-9]*.zip\)".*/\1/p' \
      | head -n 1)"
    ;;
  Darwin-x86_64)
    PLINK_URL="$(curl -fsSL https://www.cog-genomics.org/plink/2.0/ \
      | sed -n 's/.*href="\(https:\/\/s3.amazonaws.com\/plink2-assets\/plink2_mac_[0-9]*.zip\)".*/\1/p' \
      | head -n 1)"
    ;;
  Linux-x86_64)
    PLINK_URL="$(curl -fsSL https://www.cog-genomics.org/plink/2.0/ \
      | sed -n 's/.*href="\(https:\/\/s3.amazonaws.com\/plink2-assets\/plink2_linux_x86_64_[0-9]*.zip\)".*/\1/p' \
      | head -n 1)"
    ;;
  Linux-aarch64)
    PLINK_URL="$(curl -fsSL https://www.cog-genomics.org/plink/2.0/ \
      | sed -n 's/.*href="\(https:\/\/s3.amazonaws.com\/plink2-assets\/plink2_linux_aarch64_[0-9]*.zip\)".*/\1/p' \
      | head -n 1)"
    ;;
  *)
    echo "error: unsupported platform ${OS}-${ARCH}" >&2; exit 1 ;;
esac

if [[ -z "${PLINK_URL:-}" ]]; then
  echo "error: could not resolve plink2 download URL for ${OS}-${ARCH}" >&2; exit 1
fi

curl -fsSL "${PLINK_URL}" -o "${TMP_DIR}/plink2.zip"
unzip -q -o "${TMP_DIR}/plink2.zip" -d "${TMP_DIR}/plink2"
install -m 0755 "$(find "${TMP_DIR}/plink2" -type f -name plink2 | head -n 1)" "${BIN_DIR}/plink2"
"${BIN_DIR}/plink2" --version

# ---------- ADMIXTURE ----------
echo "[2/2] Installing ADMIXTURE..."

case "${OS}" in
  Darwin)
    if [[ "${ARCH}" == "arm64" ]]; then
      if ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
        echo "error: Rosetta required for ADMIXTURE on Apple Silicon." >&2
        echo "Install: softwareupdate --install-rosetta --agree-to-license" >&2
        exit 1
      fi
    fi
    ADMIX_URL="https://dalexander.github.io/admixture/binaries/admixture_macosx-1.3.0.tar.gz"
    ;;
  Linux)
    ADMIX_URL="https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz"
    ;;
  *)
    echo "error: unsupported OS ${OS} for ADMIXTURE" >&2; exit 1 ;;
esac

curl -fsSL "${ADMIX_URL}" -o "${TMP_DIR}/admixture.tar.gz"
tar -xzf "${TMP_DIR}/admixture.tar.gz" -C "${TMP_DIR}"
install -m 0755 "$(find "${TMP_DIR}" -type f -name admixture | head -n 1)" "${BIN_DIR}/admixture"

# Verify — on macOS ARM64, ADMIXTURE needs Rosetta
if [[ "${OS}" == "Darwin" && "${ARCH}" == "arm64" ]]; then
  /usr/bin/arch -x86_64 "${BIN_DIR}/admixture" --help >/dev/null 2>&1
else
  "${BIN_DIR}/admixture" --help >/dev/null 2>&1
fi

echo ""
echo "Installed:"
echo "  plink2:    ${BIN_DIR}/plink2"
echo "  admixture: ${BIN_DIR}/admixture"
echo ""
echo "Add to PATH:  export PATH=\"${BIN_DIR}:\$PATH\""
