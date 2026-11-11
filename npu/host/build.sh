#!/usr/bin/env bash
# Builds liblfm3_npu.so against a configured and built hrx-system tree.
# Usage: build.sh <hrx-source> <hrx-build> <output-dir>
# The build tree must contain loom-test (it links the same objects).
set -euo pipefail
S=$(realpath "$1"); B=$(realpath "$2"); OUT=$(realpath -m "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cc=$(python3 - "$B" <<'PY'
import json, shlex, sys
for e in json.load(open(sys.argv[1] + '/compile_commands.json')):
    if e['file'].endswith('tools/loom-test/loom-test.c'):
        args = shlex.split(e['command'])
        keep = [a for a in args[1:] if a.startswith(('-D', '-I')) or a == '-isystem' or (args[args.index(a)-1] == '-isystem' if a in args else False)]
        out, skip = [], False
        for i, a in enumerate(args[1:], 1):
            if skip: out.append(a); skip = False; continue
            if a == '-isystem': out.append(a); skip = True; continue
            if a.startswith(('-D', '-I')): out.append(a)
        print(args[0] + ' ' + ' '.join(shlex.quote(a) for a in out))
        break
PY
)
compiler=${cc%% *}
flags=${cc#* }
# Objects and libraries of loom-test's link, minus its main object.
libs=$(awk '/^build loom\/src\/loom\/tools\/loom-test\/loom-test:/{f=1;next} f&&/LINK_LIBRARIES/{sub(/^ *LINK_LIBRARIES = /,""); print; exit}' "$B/build.ninja")
objs=$(awk '/^build loom\/src\/loom\/tools\/loom-test\/loom-test:/{for(i=1;i<=NF;i++) if($i ~ /\.o$/ && $i !~ /loom-test\.c\.o$/) print $i; exit}' "$B/build.ninja")
eval "$compiler $flags -O2 -fPIC -fvisibility=hidden -std=gnu11 -c '$HERE/lfm3_npu.c' -o '$OUT/lfm3_npu.o'"
cd "$B"
eval "${compiler/clang/clang++} -shared -fuse-ld=lld -pthread -o '$OUT/liblfm3_npu.so' '$OUT/lfm3_npu.o' $libs ${EXTRA_LDFLAGS:-} -ldl -lrt -lm"
echo "$OUT/liblfm3_npu.so"
