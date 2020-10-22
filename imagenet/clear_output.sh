#!/bin/bash

# set color, see https://stackoverflow.com/a/20983251/5634636
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
RESET=$(tput sgr0)

NAME=$1
echo "Removing the output of ${NAME}"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

if [[ ! -d "out/${NAME}" ]]
then
    echo "ERROR! ${NAME} do not exists. "
    exit 1
fi

echo "${RED}Files and directories will be removed:"
ls "./out/${NAME}/"
ls "./log/${NAME}/"
ls "./logs/${NAME}/"
ls "./logs/imagenet_ckpt/${NAME}"
ls "./ckpt/${NAME}"
ls "./terminal/${NAME}"

read -p "Are you sure? ${RESET}" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # do dangerous stuff
    rm -rf out/${NAME}/
    rm -rf log/${NAME}/
    rm -rf logs/${NAME}/
    rm -rf logs/imagenet_ckpt/${NAME}
    rm -rf "./ckpt/${NAME}"
    rm -rf "./terminal/${NAME}"
fi