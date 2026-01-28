# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    16384
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        minicpm-4-1B-infllmv2)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-1B-locret)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-1B-dma)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-1B-s-dma)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-1B-nosa)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-8B-ed-dma)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-8B-ed-dma-r)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
        minicpm-4-8B-nosa)
            MODEL_PATH="/yourpath"
            MODEL_TEMPLATE_TYPE="minicpm4"
            MODEL_FRAMEWORK="hf"
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
