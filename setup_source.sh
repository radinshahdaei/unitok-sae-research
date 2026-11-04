#!/usr/bin/env bash
# Explicit small source download; never downloads model checkpoints or datasets.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
checkout_dir="$project_dir/external/unitok-sae"
pinned_revision=3a22e697e8c6586e8c270808458f17b24d8a575f
if [[ -e "$checkout_dir" ]]; then
    [[ "$(git -C "$checkout_dir" rev-parse HEAD)" == "$pinned_revision" ]]
    printf '%s\n' "Existing pinned source retained: $checkout_dir"
else
    mkdir -p "$project_dir/external"
    git clone --depth 1 --filter=blob:none --no-checkout \
        https://github.com/roronoazoero/unitok-sae.git "$checkout_dir"
    if [[ "$(git -C "$checkout_dir" rev-parse HEAD)" != "$pinned_revision" ]]; then
        git -C "$checkout_dir" fetch --depth 1 origin "$pinned_revision"
    fi
    git -C "$checkout_dir" sparse-checkout set --no-cone \
        '/*' '!/assets/' '!/wandb/' '!*.ipynb' '!/eval/llamagen/evaluations/t2i/coco_captions.csv'
    git -C "$checkout_dir" checkout --detach "$pinned_revision"
fi
# Apply recorded patches idempotently; refuse unrelated local changes.
for patch in "$project_dir"/unitok_sae/patches/*.patch; do
    [[ -e "$patch" ]] || continue
    if git -C "$checkout_dir" apply --check "$patch" 2>/dev/null; then
        git -C "$checkout_dir" apply "$patch"
    elif ! git -C "$checkout_dir" apply --check --reverse "$patch" 2>/dev/null; then
        printf 'Patch cannot be applied cleanly (unrelated changes present?): %s\n' "$patch" >&2
        exit 1
    fi
done
printf '%s\n' "Pinned source ready (with recorded patches): $checkout_dir"
