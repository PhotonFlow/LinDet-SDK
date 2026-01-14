#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f "/lindet_ws/install/setup.bash" ]; then
  source /lindet_ws/install/setup.bash
fi

exec "$@"