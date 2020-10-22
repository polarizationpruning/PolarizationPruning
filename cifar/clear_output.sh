#!/bin/bash

# set color, see https://stackoverflow.com/a/20983251/5634636
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
RESET=$(tput sgr0)

function list_file() {
  local DIR_NAME=$1
  local NAME=$2 # experiment name
  ls -d ./${DIR_NAME}/${NAME}/*
  echo
}

function remove_output() {
  local NAME=$1
  echo "Removing the output of ${NAME}"

  if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit 1
  fi

  if [[ ! -d "terminal/${NAME}" && ! -d "terminal/${NAME}_1" ]]; then
    echo "ERROR! ${NAME} do not exists. "
    exit 1
  fi

  echo "${RED}Files and directories will be removed:"
  list_file "out" "${NAME}"
  list_file "event" "${NAME}"
  list_file "ckpt" "${NAME}"
  list_file "terminal" "${NAME}"
  list_file "backup" "${NAME}"

  read -p "Are you sure? ${RESET}" -n 1 -r
  echo # (optional) move to a new line
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    # do dangerous stuff
    rm -rf "./out/${NAME}/"
    rm -rf "./event/${NAME}/"
    rm -rf "./backup/${NAME}/"
    rm -rf "./ckpt/${NAME}"
    rm -rf "./terminal/${NAME}"
  fi
  echo
  echo
}

BASE_NAME=$1
remove_output "${BASE_NAME}"
