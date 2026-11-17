#!/usr/bin/env bash
# smoke17.sh <шаг> [n] : проверка нового роутера на живой карте, 17.11.
#   prep    зависимости проекта, модель с Hugging Face, сверка хэша
#   strict  малый 0a: разрезы и сквозные тесты через роутер, инвариантность включена
#   fast    малый 0c: то же без инвариантности
set -uo pipefail
W=/workspace; VENV=$W/venv; PROJ=$W/undead
export HF_HOME=/dev/shm/hf
unset UV_NO_CACHE; export UV_CACHE_DIR=/dev/shm/uv-cache UV_LINK_MODE=copy
export PATH=$VENV/bin:$PATH
REPO=Qwen/Qwen2.5-1.5B-Instruct
REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
SHA=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
M=/dev/shm/models/Qwen2.5-1.5B-Instruct
cd "$PROJ" || exit 1
N=${2:-12}
case "${1:-}" in
prep)
  uv pip install --python "$VENV/bin/python" -q -r requirements.txt || exit 11
  mkdir -p "$M"
  if command -v hf >/dev/null; then
    hf download "$REPO" --revision "$REV" --local-dir "$M" >/dev/null || exit 12
  else
    huggingface-cli download "$REPO" --revision "$REV" --local-dir "$M" >/dev/null || exit 12
  fi
  got=$(sha256sum "$M/model.safetensors" | cut -d' ' -f1)
  echo "model.safetensors sha256 $got"
  if [ "$got" = "$SHA" ]; then echo "хэш совпал с настоящим Hugging Face"; else echo "ХЭШ НЕ СОВПАЛ"; exit 13; fi
  ls "$M" | tr '\n' ' '; echo; du -sh "$M" | cut -f1
  sha256sum router.py step0.py | cut -c1-16,65-
  ;;
strict)
  python step0.py run 0a --model "$M" --only-config bi-cache-off --n "$N" --e2e-prompts "$N" \
    --kills 10 --stops 4 --drops 4 --batch 8 --repeats 4 --router-port 8090 --out results-17-strict
  ;;
fast)
  python step0.py run 0c --model "$M" --n "$N" --e2e-prompts "$N" \
    --kills 8 --stops 3 --drops 3 --batch 8 --repeats 4 --router-port 8090 --out results-17-fast
  ;;
*)
  echo "smoke17.sh <prep|strict|fast> [n]"; exit 2
  ;;
esac
