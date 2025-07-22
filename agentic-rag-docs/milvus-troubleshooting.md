# Milvus Troubleshooting

When using the self-hosted Milvus container you may encounter startup or connection errors. The table below collects the most common checks.

| Issue | How to Verify / Fix |
|-------|--------------------|
| **MinIO not reachable** | Ensure a MinIO service is running and reachable from inside the container. Run `curl http://localhost:9000/minio/health/live` from the Milvus container. If the request fails, start or restart MinIO. |
| **Services starting too quickly** | Increase retry and backoff values in the Milvus configuration files (for example `rootCoord.grpcConnectionPool.rcpInitMaxRetry` and the `backoff` options). This allows components more time to find dependencies. |
| **etcd out of sync** | Verify etcd health with `etcdctl endpoint health` and inspect the etcd logs. All endpoints must report `healthy`. |
| **Configuration mismatch** | Double-check that every component references the same etcd endpoints and RootCoord settings. In `compose.yaml` these are set with the `ETCD_ENDPOINTS` environment variable. |

Start by confirming all services are running with `docker ps` and inspect individual logs with `docker logs <service>`. After applying the changes, restart the stack using `docker compose up -d`.
