#!/bin/bash
# Submit Spark job to YARN cluster instead of local mode.
# Demonstrates distributed execution on the Hadoop ecosystem.
#
# Usage:
#   bash pipelines/yarn_submit.sh

set -e

export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop

echo "============================================"
echo " Submitting Spark job to YARN"
echo "============================================"
echo "  Master:     yarn"
echo "  Deploy:     client"
echo "  Driver:     1g"
echo "  Executors:  1 × 1g"
echo "============================================"

spark-submit \
    --master yarn \
    --deploy-mode client \
    --driver-memory 1g \
    --executor-memory 1g \
    --num-executors 1 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
    --conf spark.yarn.submit.waitAppCompletion=true \
    pipelines/hdfs_pipeline.py

echo ""
echo "============================================"
echo " ✅ Job submitted. Check progress at:"
echo "    YARN RM UI:  http://localhost:8088"
echo "    History:     http://localhost:19888"
echo "============================================"