#!/usr/bin/env bash
source .env
source "$PYTHON_ENV/bin/activate"

python __main__.py \
  --profiles_json "${BASE_PROJECT_PATH}/sh/profiles.json" \
  --N 50 --log_scale