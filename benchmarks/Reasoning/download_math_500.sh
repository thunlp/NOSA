#!/bin/bash

URL="https://media.githubusercontent.com/media/openai/prm800k/refs/heads/main/prm800k/math_splits/test.jsonl?download=true"
OUTPUT="math-500.jsonl"

curl -L "$URL" -o "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "success"
else
    echo "fail"
fi
