FROM milvusdb/milvus:v2.5.8 as milvus
FROM quay.io/coreos/etcd:v3.5.18 as etcd
FROM minio/minio:RELEASE.2023-03-20T20-16-18Z as minio

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y supervisor curl && rm -rf /var/lib/apt/lists/*

# copy binaries from stage images
COPY --from=milvus / /milvus-base
COPY --from=etcd /usr/local/bin/etcd /usr/local/bin/etcd
COPY --from=etcd /usr/local/bin/etcdctl /usr/local/bin/etcdctl
COPY --from=minio /usr/bin/minio /usr/local/bin/minio
RUN chmod +x /usr/local/bin/minio

# set up directories
RUN mkdir -p /var/lib/milvus /var/lib/etcd /data/minio

# link milvus install
RUN ln -s /milvus-base/opt/milvus /opt/milvus

COPY init.sh /init.sh
COPY supervisord.conf /etc/supervisor/conf.d/milvus.conf
RUN chmod +x /init.sh

EXPOSE 19530 19531 9091 9000 9001
HEALTHCHECK --interval=30s --start-period=60s --timeout=5s CMD curl -f http://localhost:9091/healthz || exit 1

CMD ["/init.sh"]
