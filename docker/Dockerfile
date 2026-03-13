FROM dustynv/ros:humble-pytorch-l4t-r32.7.1 AS base
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update || true && \
    apt-get install -y --no-install-recommends \
        curl gnupg2 nano git \
        python3-pip \
        v4l-utils \
        # GStreamer development
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        # OpenCV (L4T pre-built)
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    opencv-python-headless \
    pycuda

WORKDIR /lindet_ws

# ═══════════════════════════════════════════════════════════════════════════════
# Builder stage — colcon build
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY ./src /lindet_ws/src

# Install ROS dependencies
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y || true

# Build all packages
RUN . /opt/ros/humble/setup.sh && \
    colcon build \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-ignore lindet_detection lindet_camera || true

# Build C++ packages (may fail without TRT/GStreamer on non-Jetson)
RUN . /opt/ros/humble/setup.sh && \
    colcon build \
    --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select lindet_camera lindet_detection || true

# ═══════════════════════════════════════════════════════════════════════════════
# Runner stage — minimal runtime image
# ═══════════════════════════════════════════════════════════════════════════════
FROM base AS runner

COPY --from=builder /lindet_ws/install /lindet_ws/install
COPY ./config /lindet_ws/config
COPY ./models /lindet_ws/models
COPY ./docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "lindet_bringup", "lindet_system.launch.py"]