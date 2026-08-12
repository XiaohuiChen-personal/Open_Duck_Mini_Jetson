#!/usr/bin/env bash
# Set up (or verify) the CAD/URDF toolchain this repo's geometry tasks need.
#
# WHY THIS EXISTS
# ---------------
# The `cad`, `urdf` and `cad-viewer` agent skills need a Python environment the
# system interpreter does not provide, and the failure mode is a trap: STL
# rendering works on plain `python3`, so the toolchain LOOKS fine right up until
# something touches a STEP file, which then dies with `No module named 'OCP'`.
# STEP is exactly what task_plan_v2.md Task M3 produces.
#
#   system python3        playwright yes, OCP no   -> .stl renders, .step FAILS
#   ~/.venvs/cad-viewer   cadgen + OCP + build123d + playwright -> both work
#
# Two more traps this script exists to absorb:
#
#   * `npm run start` for cad-viewer is BROKEN in build 0.4.5 — package.json's
#     start script runs `node scripts/start-viewer.mjs`, a file that does not
#     exist in the package, and package.json declares zero dependencies, so
#     `npm install` fixes nothing. The viewer is a Python backend.
#   * Ubuntu 24.04 refuses a system-wide `pip install` under PEP 668. Hence a
#     venv. Do NOT pass --break-system-packages.
#
# The venv deliberately lives OUTSIDE the skill directory. Skills are installed
# from github.com/earthtojake/text-to-cad and tracked in ~/.agents/.skill-lock.json,
# so ~/.agents/skills/<name>/ is replaced wholesale on update — anything stored
# there is lost silently, including a venv.
#
# USAGE
#   scripts/setup_cad_tools.sh           # install what is missing, then verify
#   scripts/setup_cad_tools.sh --check   # verify only, change nothing
#
# Exits 0 when the toolchain is usable, nonzero otherwise, so it works as a gate
# at the top of any geometry task.

set -uo pipefail

VENV="${CAD_VENV:-$HOME/.venvs/cad-viewer}"
PY="$VENV/bin/python3"
SKILLS="${CAD_SKILLS_DIR:-$HOME/.claude/skills}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
info() { printf '  ...   %s\n' "$1"; }
FAILED=0

echo "CAD toolchain — venv: $VENV"
echo

# ---------------------------------------------------------------- 1. the venv
if [ ! -x "$PY" ]; then
  if [ "$CHECK_ONLY" = 1 ]; then
    bad "venv missing at $VENV (run without --check to create it)"; exit 1
  fi
  info "creating venv"
  python3 -m venv "$VENV" || { bad "could not create venv"; exit 1; }
fi
ok "venv interpreter $($PY --version 2>&1)"

# ------------------------------------------------------- 2. python packages
# The cad skill's own requirements.txt is `cadgen==0.4.5` + `playwright`.
# cadgen is what drags in OCP (the OpenCascade binding) and build123d.
need_pkgs=0
for m in cadgen OCP build123d playwright; do
  "$PY" -c "import $m" 2>/dev/null || need_pkgs=1
done

if [ "$need_pkgs" = 1 ]; then
  if [ "$CHECK_ONLY" = 1 ]; then
    bad "python packages missing (run without --check to install)"; FAILED=1
  else
    REQ="$SKILLS/cad/requirements.txt"
    if [ -f "$REQ" ]; then
      info "installing from $REQ"
      "$VENV/bin/pip" install -q -r "$REQ" || { bad "pip install failed"; exit 1; }
    else
      info "skill requirements.txt not found; installing known-good set"
      "$VENV/bin/pip" install -q 'cadgen==0.4.5' playwright || { bad "pip install failed"; exit 1; }
    fi
  fi
fi

for m in cadgen OCP build123d playwright; do
  v="$("$PY" -c "import $m,sys; sys.stdout.write(getattr($m,'__version__','') or '')" 2>/dev/null)"
  if [ -n "$v" ] || "$PY" -c "import $m" 2>/dev/null; then
    ok "$m ${v:-installed}"
  else
    bad "$m missing"; FAILED=1
  fi
done

# ------------------------------------------------------------ 3. chromium
# playwright ships no browser; snapshot rendering needs the headless shell.
if ! "$PY" -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    p.chromium.launch(headless=True).close()
" >/dev/null 2>&1; then
  if [ "$CHECK_ONLY" = 1 ]; then
    bad "chromium not installed for playwright"; FAILED=1
  else
    info "downloading chromium (~110 MB, one time)"
    "$PY" -m playwright install chromium >/dev/null 2>&1 || { bad "chromium install failed"; FAILED=1; }
  fi
fi
"$PY" -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    p.chromium.launch(headless=True).close()
" >/dev/null 2>&1 && ok "chromium launches headless" || { bad "chromium cannot launch"; FAILED=1; }

# ------------------------------------------- 4. prove a render actually works
# Deterministic checks passing is not proof the toolchain renders. Render a real
# part and assert the PNG is a plausible image, not a 0-byte stub.
SNAP="$SKILLS/cad/scripts/snapshot"
PART="$REPO/print/roll_motor_top.stl"
if [ -e "$SNAP" ] && [ -f "$PART" ]; then   # $SNAP is a package DIRECTORY, not a file
  TMP="$(mktemp -d)"
  if "$PY" "$SNAP" --input "$PART" --output "$TMP/probe.png" >/dev/null 2>&1; then
    IMG="$(find "$TMP" -name '*.png' -size +10k | head -1)"
    [ -n "$IMG" ] && ok "rendered a real STL ($(stat -c%s "$IMG") bytes)" \
                  || { bad "snapshot produced no usable PNG"; FAILED=1; }
  else
    bad "snapshot failed on $PART"; FAILED=1
  fi
  rm -rf "$TMP"
else
  bad "render probe COULD NOT RUN (snapshot=$SNAP part=$PART)"
  bad "  a skipped probe is not a pass -- the render is the check that matters"
  FAILED=1
fi

# ------------------------------------------------------------------ report
echo
if [ "$FAILED" = 0 ]; then
  cat <<EOF
CAD toolchain READY.

  Always invoke the skill scripts with THIS interpreter:
      $PY $SKILLS/cad/scripts/snapshot --input <file.stl|.step> --output out.png
      $PY $SKILLS/cad/scripts/inspect refs <file.step> --facts
      $PY $SKILLS/urdf/scripts/validate <file.urdf>

  Using plain 'python3' renders .stl but FAILS on .step with "No module named 'OCP'".

  Optional live viewer (browser review, not needed for agent rendering):
      cd ~/.agents/skills/cad-viewer/scripts/viewer
      $PY -m server_py.start_viewer --port 3245
EOF
  exit 0
fi
bad "toolchain NOT ready — see failures above"
exit 1
