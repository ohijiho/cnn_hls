#!/bin/bash

left_pid=$1
right_pid=$2

for i in {0..999}; do
    left="py/dump/${left_pid}/${i}.txt"
    right="cmake-build-release/dump/${right_pid}/${i}.txt"
    echo "### Iter $i ###"
    diff "$left" "$right"
done
