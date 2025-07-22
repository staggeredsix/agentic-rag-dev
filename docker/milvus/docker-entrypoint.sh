#!/usr/bin/env bash
set -e
MAX_RETRIES=30

wait_for_port() {
  local port="$1"
  local name="$2"
  for i in $(seq ${MAX_RETRIES}); do
    if nc -z localhost "$port" >/dev/null 2>&1; then
      echo "$name healthy"
      return 0
    fi
    echo "waiting for $name on port $port..."
    sleep 1
  done
  echo "timed out waiting for $name" >&2
  exit 1
}

# start etcd
etcd --data-dir /var/lib/etcd \
     --advertise-client-urls http://0.0.0.0:2379 \
     --listen-client-urls http://0.0.0.0:2379 &
ETCD_PID=$!
wait_for_port 2379 "etcd"

# start minio
if [ -f /etc/minio.env ]; then
  source /etc/minio.env
fi
minio server /data/minio --console-address :9001 &
MINIO_PID=$!
wait_for_port 9000 "MinIO"

export ETCD_ENDPOINTS=http://127.0.0.1:2379
export MINIO_ADDRESS=127.0.0.1:9000

# start milvus components
milvus run rootCoord &
wait_for_port 53100 "RootCoord"

milvus run dataCoord &
wait_for_port 13333 "DataCoord"

milvus run queryCoord &
wait_for_port 19531 "QueryCoord"

milvus run indexCoord &
wait_for_port 31000 "IndexCoord"

milvus run queryNode &
wait_for_port 21123 "QueryNode"

milvus run dataNode &
wait_for_port 21124 "DataNode"

milvus run indexNode &
wait_for_port 21121 "IndexNode"

milvus run proxy &
wait_for_port 19530 "Proxy"

# wait for any process to exit
wait -n
