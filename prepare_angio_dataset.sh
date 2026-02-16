#!/bin/bash

set -e
source .env

mkdir -p datasets
mkdir -p datasets/angio
mkdir -p datasets/angio/train
mkdir -p datasets/angio/test
mkdir -p datasets/angio/train/public
mkdir -p datasets/angio/test/public
mkdir -p datasets/angio/train/private
mkdir -p datasets/angio/test/private

dst_public_path_train=datasets/angio/train/public
dst_public_path_test=datasets/angio/test/public
dst_private_path_train=datasets/angio/train/private
dst_private_path_test=datasets/angio/test/private

if [[ -z "$src_public_path_train" || -z "$src_public_path_test" || -z "$src_private_path_train" || -z "$src_private_path_test" ]]; then
  echo "Paths environment variables must be set."
  exit 1
fi

cp -r "$src_public_path_train"/* "$dst_public_path_train"/
cp -r "$src_public_path_test"/* "$dst_public_path_test"/
cp "$src_private_path_train"/_*[1-9].png "$dst_private_path_train"/ || true
cp "$src_private_path_test"/_*[1-9].png "$dst_private_path_test"/ || true

uv run remove_excess_files.py --target_dir $dst_public_path_train --num_images 228
uv run remove_excess_files.py --target_dir $dst_public_path_test --num_images 132