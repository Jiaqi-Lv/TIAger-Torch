#!/usr/bin/env bash

./build.sh

docker save tiager_torch | gzip -c > tiager_torch.tar.gz