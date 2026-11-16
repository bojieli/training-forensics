#!/bin/bash
# Fix-E: score runs with reader v2 (default TF_READER=v2) into results/fixe/<set>/.
# usage: scripts/fixe/rescore.sh <set> <run-dir>...   (TF_PAR parallel jobs, TF_PROCS per job)
cd "$(dirname "$0")/../.."
set_=$1; shift
mkdir -p results/fixe/$set_ results/fixe/logs
printf "%s\n" "$@" | xargs -P "${TF_PAR:-4}" -I{} sh -c '
  r={}; n=$(basename $r)
  case $r in runs/stacks/*) s=scripts/stacks/score_stack.py; extra="";;
              *) s=scripts/controlled/score.py; extra="--outdir results/fixe/'"$set_"'";; esac
  if [ "$s" = scripts/stacks/score_stack.py ]; then
    TF_OUTDIR=results/fixe/'"$set_"' python3 -u $s $r > results/fixe/logs/'"$set_"'_$n.log 2>&1
  else
    python3 -u $s $r $extra > results/fixe/logs/'"$set_"'_$n.log 2>&1
  fi
  echo "done $n $?"'
