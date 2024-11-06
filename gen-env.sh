#!/bin/bash

# if error, exit with a failure return code
set -e pipefail

ENV_GEN_FILE=".env"

# Overwrite existing file
printf "#\n# Do not edit this file. It is generated!\n#\n" > tmp_env_file

env_files=$(find . -name "*.env" -not -path "./.env")

# Concatenate all the ".env" files into a single file called ".env"
cat tmp_env_file $env_files > $ENV_GEN_FILE

rm tmp_env_file