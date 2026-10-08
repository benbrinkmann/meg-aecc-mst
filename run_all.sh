#!/usr/bin/env bash
# Run the per-subject stages in order, then the group stage.
# Usage:  ./run_all.sh                 all rows in subjects.csv
#         ./run_all.sh --subject P001  one subject (all its sessions)
set -euo pipefail
cd "$(dirname "$0")"

for stage in s01_mri_prep s02_coreg s03_atlas_forward s04_beamformer s05_connectivity s06_mst; do
    echo "##### ${stage} #####"
    python3 "${stage}.py" "$@"
done

# The group stage always uses every row in subjects.csv.
if [ "$#" -eq 0 ]; then
    echo "##### s07_group_reference #####"
    python3 s07_group_reference.py
fi
