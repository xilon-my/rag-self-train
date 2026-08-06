#!/usr/bin/env bash
# Fetch FinRAGBench-V Chinese financial PDFs (pdfs_for_QA/pdf_ch.tar.gz, ~4.5GB).
# License: apache-2.0 (verify via HF metadata). PDFs are never committed to git.
set -euo pipefail

DATASET="zhaosuifeng/FinRAGBench-V"
FILE="pdfs_for_QA/pdf_ch.tar.gz"
MIRROR_URL="https://hf-mirror.com/datasets/${DATASET}/resolve/main/${FILE}"
CORPUS_DIR="${1:-corpus}"
mkdir -p "${CORPUS_DIR}"

echo "downloading ${MIRROR_URL} -> ${CORPUS_DIR}/pdf_ch.tar.gz"
curl -L --retry 5 --retry-delay 5 -o "${CORPUS_DIR}/pdf_ch.tar.gz" "${MIRROR_URL}"

echo "extracting..."
tar -xzf "${CORPUS_DIR}/pdf_ch.tar.gz" -C "${CORPUS_DIR}"
echo "done. contents:"
find "${CORPUS_DIR}" -maxdepth 2 -type d | head -20
