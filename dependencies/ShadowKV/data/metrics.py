################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str

def string_match_part(preds, refs):
    preds = postprocess_pred(preds)
    if isinstance(refs, str):
        refs = [refs]
    score_ref_in_pred = max([1.0 if r.lower() in preds.lower() else 0.0 for r in refs])
    score_pred_in_ref = max([1.0 if preds.lower() in r.lower() else 0.0 for r in refs])
    score = max(score_ref_in_pred, score_pred_in_ref)
    return round(score, 2)

def multi_number(prediction: str, ground_truth: list) -> float:
    assert type(prediction) == str, f"Prediction is not a string, but {prediction}, type: {type(prediction)}"
    assert type(ground_truth) == list, f"Ground truth is not a list, but {ground_truth}, type: {type(ground_truth)}"
    prediction = normalize_answer(prediction)
    prediction_list = re.findall(r'\d+', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    hit_rate = len(hits) / len(ground_truth)
    
    return hit_rate

def multi_words(prediction: str, ground_truth: list) -> float:
    prediction = prediction.lower()
    ground_truth = [gt.lower() for gt in ground_truth]
    prediction_list = re.findall(r'\b\w+\b', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    hit_rate = len(hits) / len(ground_truth)
    
    return hit_rate

def needle_score(prediction, ground_truth):
    assert type(prediction) == str, f"Prediction is not a string, but {prediction}, type: {type(prediction)}"
    assert type(ground_truth) == str, f"Ground truth is not a string, but {ground_truth}, type: {type(ground_truth)}"
    prediction = normalize_answer(postprocess_pred(prediction))
    ground_truth = normalize_answer(ground_truth)
    min_length = min(len(prediction), len(ground_truth))
    min_length = len(ground_truth)
    score =  float((prediction[:min_length] == ground_truth[:min_length]))
    pred_list = prediction.split()
    score = max(float(ground_truth in pred_list), score)
    return score
