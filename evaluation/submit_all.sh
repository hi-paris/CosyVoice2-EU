#!/usr/bin/env bash
set -euo pipefail

# ---------- knobs ----------
PAIR_SIZE=1                           # 2 -> (50,100) per job; set to 1 for single-hour jobs (24 jobs)
ADD_LANG_HINT=true                    # pass --add-language-hint to your pipeline
A100_PARTITION="A100"                 # change if your site uses another name
L40S_PARTITION="L40S"                 # change if your site uses another name
A100_GRES="gpu:1"                     # e.g. "gpu:a100:1" if your cluster is strict
L40S_GRES="gpu:1"                     # e.g. "gpu:l40s:1"
MAIL_USER="tim.horstmann@ip-paris.fr" # used on sbatch submission line
# -------------------------------------

RUN_SCRIPT="run_eval.sh"              # your existing SLURM job script
# HOURS=(50 100 250 500 1000 1500)
HOURS=(100 500 1500)
COMBOS=( "fr|false" "de|false" "fr|true" "de|true" )  # lang|use_mixed_model

# build hour groups based on PAIR_SIZE
pairs=()
labels=()
i=0
while [ $i -lt ${#HOURS[@]} ]; do
  chunk=("${HOURS[@]:$i:$PAIR_SIZE}")
  if [ ${#chunk[@]} -eq 0 ]; then break; fi
  pairs+=( "$(IFS=,; echo "${chunk[*]}")" )
  # label like 1-2, 3-4, ...
  start=$((i+1))
  end=$((i+${#chunk[@]}))
  labels+=( "${start}-${end}" )
  i=$((i+PAIR_SIZE))
done

# build job list: each entry is "lang|mix|label|hours_csv"
jobs=()
for combo in "${COMBOS[@]}"; do
  IFS='|' read -r LANG MIX <<< "$combo"
  for idx in "${!pairs[@]}"; do
    jobs+=( "${LANG}|${MIX}|${labels[$idx]}|${pairs[$idx]}" )
  done
done

total=${#jobs[@]}
# how many to A100 (rounded)
a100_target=$(( (total * 60 + 50) / 100 ))

echo "Planning ${total} jobs -> ${a100_target} on ${A100_PARTITION}, $((total-a100_target)) on ${L40S_PARTITION}"

# submit with approx 60/40 spread across the sequence
a100_count=0
for jidx in "${!jobs[@]}"; do
  IFS='|' read -r LANG MIX LABEL HOURS_CSV <<< "${jobs[$jidx]}"

  # choose partition: keep proportion by index
  # (spread: first ceil(60%) A100, remaining L40S)
#   if [ $jidx -lt $a100_target ]; then
    PART="$A100_PARTITION";  GRES="$A100_GRES"
#   else
    # PART="$L40S_PARTITION";  GRES="$L40S_GRES"
#   fi

  # job-name like: m-fr-1-2 or fr-3-4
  PREFIX=""
  if [[ "$MIX" == "true" ]]; then PREFIX="m-"; fi
  JOB_NAME="${PREFIX}${LANG}-${LABEL}"

  # submit
  echo "Submitting ${JOB_NAME} (${LANG}, mix=${MIX}, hours=${HOURS_CSV}, lang_hint=${ADD_LANG_HINT}) -> ${PART}"
  LANGUAGE_OVERRIDE="${LANG}" \
  HOURS_OVERRIDE="${HOURS_CSV}" \
  USE_MIXED_MODEL="${MIX}" \
  ADD_LANG_HINT="${ADD_LANG_HINT}" \
  sbatch  \
    --job-name="${JOB_NAME}" \
    --partition="${PART}" \
    --gres="${GRES}" \
    --output="logs/${JOB_NAME}-%j.out" \
    --error="logs/${JOB_NAME}-%j.err" \
    --mail-type=BEGIN,END,FAIL \
    --mail-user="${MAIL_USER}" \
    --export=ALL \
    "${RUN_SCRIPT}"
done