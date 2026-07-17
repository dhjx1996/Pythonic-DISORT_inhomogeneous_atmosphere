#!/bin/bash
# Helper for the strong-correlation pipeline (AGENT_strongcorr_pipeline.md).
# Usage:
#   pipeline_helper.sh missing <parts_dir>            -> comma list of idx 1..125 lacking _A.npz (empty if complete)
#   pipeline_helper.sh count   <parts_dir>            -> "<done>/125"
#   pipeline_helper.sh bundle  <parts_dir> <logs_dir> <bundle_basename>  -> zip to workspace root
ROOT=/burg-archive/home/dh3065/cloud_profile_retrieval/Pythonic-DISORT_inhomogeneous_atmosphere
WS=/burg-archive/home/dh3065/cloud_profile_retrieval
cmd=$1
case "$cmd" in
  missing)
    d=$2; miss=""
    for i in $(seq 1 125); do [ -f "$d/${i}_A.npz" ] || miss="$miss,$i"; done
    echo "${miss#,}" ;;
  count)
    d=$2; echo "$(ls $d/*_A.npz 2>/dev/null | wc -l)/125" ;;
  bundle)
    d=$2; l=$3; name=$4
    tmp=/tmp/${name}_$$; rm -rf "$tmp"; mkdir -p "$tmp"
    cp -r "$d" "$tmp/"; mkdir -p "$tmp/logs"; cp "$l"/*.out "$tmp/logs/" 2>/dev/null
    ( cd /tmp && zip -rq "${name}.zip" "$(basename $tmp)" )
    mv "/tmp/${name}.zip" "$WS/${name}.zip"; rm -rf "$tmp"
    echo "bundle -> $WS/${name}.zip ($(du -h $WS/${name}.zip | cut -f1)); npz=$(cd $d && ls *_A.npz 2>/dev/null | wc -l)" ;;
  *) echo "unknown: $cmd" ; exit 1 ;;
esac
