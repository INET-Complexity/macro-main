#!/bin/bash

# Flatten synthetic_* documentation directories
for d in ./docs/macro_data/api/processing/synthetic_*; do
  if [ -d "$d" ] && [ -f "$d/index.md" ]; then
    echo "Flattening $d ..."
    mv "$d/index.md" "${d%/}.md"
    rmdir "$d"
  fi
done

echo "Flattening complete." 