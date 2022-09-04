#/usr/bin/env bash
set -eu

commands=(
  "./bin/gen_keys -N 16384 -L 5 --q0 50 --qi 30 --ql 60 --prefix N16384_L5_50-30-60 --dataset mnist"
  "./bin/gen_keys -N 16384 -L 7 --q0 50 --qi 30 --ql 60 --prefix N16384_L7_50-30-60 --dataset mnist"
  # "./bin/gen_keys -N 16384 -L 7 --q0 50 --qi 30 --ql 60 --prefix N16384_L7_50-30-60 --dataset cifar-10"
  # "./bin/gen_keys -N 16384 -L 10 --q0 50 --qi 30 --ql 60 --prefix N16384_L10_50-30-60 --dataset cifar-10"
  "./bin/gen_keys -N 16384 -L 8 --q0 50 --qi 30 --ql 60 --prefix N16384_L8_50-30-60 --dataset cifar-10"
  "./bin/gen_keys -N 16384 -L 11 --q0 50 --qi 30 --ql 58 --prefix N16384_L11_50-30-58 --dataset cifar-10"
)

for command in "${commands[@]}"
do
  eval "${command}"
done
