#!/usr/bin/env bash
# Run the per-subject stages in order, then the group stage.
# Usage:  ./run_all.sh                 all rows in subjects.csv
#         ./run_all.sh --subject P001  one subject (all its sessions)
set -euo pipefail
cd "$(dirname "$0")"

# Python interpreter: use $PYTHON if set (e.g. PYTHON=/usr/bin/python3.11 ./run_all.sh),
# otherwise python3.11 if installed, otherwise python3.
PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3 || true)}"
if [ -z "${PYTHON}" ]; then
    echo "No Python interpreter found; set PYTHON." >&2
    exit 1
fi
echo "Using ${PYTHON}"

for stage in s01_mri_prep s02_coreg s03_atlas_forward s04_beamformer s05_connectivity s06_mst; do
    echo "##### ${stage} #####"
    "${PYTHON}" "${stage}.py" "$@"
done

# The group stage always uses every row in subjects.csv.
if [ "$#" -eq 0 ]; then
    echo "##### s07_group_reference #####"
    "${PYTHON}" s07_group_reference.py
fi

# Hub analysis: per session hub measures, then group contrasts on all subjects.
echo "##### s08_hubs #####"
"${PYTHON}" s08_hubs.py "$@"

# Subjects or sessions that failed at any stage (they were skipped, not fatal).
echo "##### Failure summary #####"
"${PYTHON}" -c "import common; common.print_failure_summary()"
