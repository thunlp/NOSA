#!/bin/bash

URL="https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
OUTPUT="data.zip"

curl -L "$URL" -o "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "Download success"
    unzip $OUTPUT
    echo "Unzip success"
else
    echo "Download failed"
    exit 1
fi
