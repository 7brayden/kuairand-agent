#!/usr/bin/env bash
# Download KuaiRand-Pure into data/raw/.
#
# Source is the Zenodo direct link published by the organisers' starter kit README
# (https://kuairand.com -> Zenodo record 10439422). No registration required.
#
# Hard rule: KuaiRand only. No external training data of any kind.
#
# Result: data/raw/KuaiRand-Pure/data/*.csv
#   log_standard_4_08_to_4_21_pure.csv   (train window)
#   log_standard_4_22_to_5_08_pure.csv   (valid + test window)
#   log_random_4_22_to_5_08_pure.csv     (unbiased random-exposure log)
#   user_features_pure.csv, video_features_basic_pure.csv, video_features_statistic_pure.csv
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR="data/raw"
URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
TARBALL="${RAW_DIR}/KuaiRand-Pure.tar.gz"

mkdir -p "${RAW_DIR}"

if [ -d "${RAW_DIR}/KuaiRand-Pure/data" ]; then
  echo "Already extracted at ${RAW_DIR}/KuaiRand-Pure/data — nothing to do."
  exit 0
fi

if [ ! -f "${TARBALL}" ]; then
  echo "Downloading KuaiRand-Pure from Zenodo ..."
  curl -fL --retry 3 --retry-delay 5 -o "${TARBALL}.part" "${URL}"
  mv "${TARBALL}.part" "${TARBALL}"
fi

echo "Extracting ..."
tar xzf "${TARBALL}" -C "${RAW_DIR}"
echo "Done: ${RAW_DIR}/KuaiRand-Pure/data"
ls -la "${RAW_DIR}/KuaiRand-Pure/data"
