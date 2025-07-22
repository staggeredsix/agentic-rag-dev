#!/usr/bin/env bash
set -e

# start etcd
etcd --data-dir /var/lib/etcd --advertise-client-urls http://127.0.0.1:2379 \
    --listen-client-urls http://0.0.0.0:2379 &

# wait for etcd to be ready
for i in {1..30}; do
  if etcdctl endpoint health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# start minio
minio server /data/minio --console-address :9001 &

# wait for minio
for i in {1..30}; do
  if curl -f http://localhost:9000/minio/health/live >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# start milvus standalone
milvus run standalone &

# wait on child processes
wait -n
