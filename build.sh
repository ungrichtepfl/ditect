#!/bin/sh

set -xe

mkdir -p ./build

CFLAGS="-Wall -Werror -Wextra -Wpedantic -Ofast -ggdb"
LFLAGS="-lm -lpthread"
CPPFLAGS=""
if [ "$(uname -m)" = "x86_64" ]; then
    RAYLIB="-I ./raylib-5.0_linux_amd64/include/ -L./raylib-5.0_linux_amd64/lib -l:libraylib.a -ldl"
else
    echo "You are not on a x86_64 machine, please install raylib 5.0.0 and make sure that pkg-config can find it."
    RAYLIB="$(pkg-config --libs --cflags "raylib")"
fi

SRC="./src/main.c"


# shellcheck disable=SC2086
cc -o ./build/dictect $SRC $CPPFLAGS $CFLAGS $RAYLIB $LFLAGS
