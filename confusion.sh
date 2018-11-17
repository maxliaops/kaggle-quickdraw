#!/usr/bin/env bash

set -e
set -o pipefail

if [ -z $1 ]
then
  echo "missing run name"
  exit 1
fi

trap archive_artifacts EXIT

function install_dependencies() {
  apt-get update >/dev/null
  apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
  rm -rf /var/lib/apt/lists/*

  pip -q install virtualenv
  virtualenv env --python=python3
  . env/bin/activate

  pip -q install -r requirements.txt
}

function archive_artifacts() {
  rm -rf /storage/models/quickdraw/${RUN_NAME}
  mkdir -p /storage/models/quickdraw/${RUN_NAME}
  cp -r /artifacts/* /storage/models/quickdraw/${RUN_NAME}
}

RUN_NAME=$1

install_dependencies

python confusion.py
