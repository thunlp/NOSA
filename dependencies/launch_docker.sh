docker run \
  -v <your_local_path>:<in_docker_path> \
  -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --network host \
  --privileged=true \
  --entrypoint bash \
  --ulimit stack=67108864 \
  "$@"
