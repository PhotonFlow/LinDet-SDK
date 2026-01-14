FROM dustynv/ros:humble-pytorch-l4t-r32.7.1 AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update || true && \
    apt-get install -y --no-install-recommends curl gnupg2 && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    git \
    python3-pip \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /lindet_ws
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

COPY ./src /lindet_ws/src
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release

FROM base AS runner
COPY --from=builder /lindet_ws/install /lindet_ws/install
COPY ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "stereo_camera_system", "system.launch.py"]