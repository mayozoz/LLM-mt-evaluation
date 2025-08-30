import requests
import json
import os
import sys
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import re
from itertools import combinations

api_key = "xxx"


def get_prompt(source, translations):
    prompt = f"""You will compare and rank 3 Tibetan translations of the same Chinese text using a 1-5 scale.
            Source (chinese): {source}
            Translation A: {translations[0]}
            Translation B: {translations[1]}
            Translation C: {translations[2]}

            Rate each translation from 1-5 based on:
            - Accuracy of translation from source
            - Fluency and naturalness of output
            - Preservation of meaning and context
            - Grammatical correctness

            Scale (whole numbers only):
            5 = Excellent, 4 = Good, 3 = Adequate, 2 = Poor, 1 = Very poor

            Consider these translations together and assign score that reflect their relative quality.
            
            Output format:
            Translation A: [score 1-5]
            Translation B: [score 1-5]
            Translation C: [score 1-5]

            Only output in the required format provided, no explainations or other text."""
    return prompt


def parse_scores_response(response_text):
    """
    Parse LLM response to extract pairwise comparisons and overall ranking
    """
    try:
        score_pattern = r'Translation ([ABC]): (\d)'
        patterns = [
            r'Translation ([ABC]):\s*(\d)',
            r'Translation ([ABC]):\s*\[(\d)\]',
            r'([ABC]):\s*(\d)',
            r'([ABC]):\s*\[(\d)\]'
        ]

        matches = re.findall(score_pattern, response_text)
        if len(matches) != 3:
            for pattern in patterns:
                other_matches = re.findall(pattern, response_text)
                if len(other_matches) == 3:
                    matches = other_matches
                    break
        scores = {}

        for match in matches:
            model = match[0]
            score = int(match[1])
            if 1 <= score <= 5:
                scores[model] = score
            else:
                print(f"Warning: Invalid score {score} for model {model}.")
        
        if len(scores) == 3 and len(matches) == 3:
            return {
                'scores': [scores.get('A', 0), scores.get('B', 0), scores.get('C', 0)],
                'raw_response': response_text,
                'parsed_successfully': True
            }
        else:
            return {
                'scores': [scores.get('A', 0), scores.get('B', 0), scores.get('C', 0)],
                'raw_response': response_text,
                'parsed_successfully': False,
                'error': f"Found {len(scores)} scores and {len(matches)}. Expected 3 and 3"
            }

    except Exception as e:
        return {
            'scores': [0, 0, 0],
            'raw_response': response_text,
            'parsed_successfully': False,
            'error': str(e)
        }


def request_gpt4(js_data):
    """
    Request evaluation from GPT-4.
    """
    time.sleep(0.1)

    source = js_data['source']
    translations = js_data['translations']
    model_names = js_data['model_names']

    prompt = get_prompt(source, translations)
    messages=[{'role': 'user', 'content':  prompt}]
    
    data = {
        "api-key": api_key,
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.0,
        "maxTokens": 3500
    }

    for i in range(3):
        try:
            response = requests.post("http://yd-chatgpt-api.inner.youdao.com/api", json=data)
            result = response.json()

            response_text = result['detail']['choices'][0]['message']['content'].strip()
            score_result = parse_scores_response(response_text)

            js_data['score_result'] = score_result
            js_data['scores'] = score_result['scores']

            if not score_result['parsed_successfully']:
                print(f"Warning: Failed to parse scores for line {js_data['id']}")
                # print(f"Error: {js_data['error']}")
                print(f"\Full response: {result}")

            return js_data

        except Exception as e:
            if i == 2:
                js_data['score_result'] = {'error': str(e)}
                js_data['scores'] = [0, 0, 0]
                print(f'#ERROR#--{str(e)}')
                print("\tFull Response:", response.json() if 'response' in locals() else str(e))

    return js_data


# Expected arguments: tibetan_file translation1_file translation2_file translation3_file
# Usage: python3 scripts/mt-ranking-gpt-4-comparative-rank.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b
# Usage: python3 scripts/mt-ranking-gpt-4-comparative-rank.py cn-tbt-200/src_clean.txt cn-tbt-200/mt-hyps/hyp_deepseek-v3 cn-tbt-200/mt-hyps/hyp_google-translate cn-tbt-200/mt-hyps/hyp_qwen2.5_72b

if len(sys.argv) != 5:
    print("Usage: python script.py <src_file> <tgt1_file> <tgt2_file> <tgt3_file>")
    sys.exit(1)

src_file, tgt1_file, tgt2_file, tgt3_file = sys.argv[1:]


with open(src_file, 'r', encoding='utf-8') as f:
    src_lines = [line.strip() for line in f.readlines()]

tgt_files = [tgt1_file, tgt2_file, tgt3_file]
tgt_lines = []
model_names = []

for tgt_file in tgt_files:
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines.append([line.strip() for line in f.readlines()]) # 2d
        model_name = os.path.basename(tgt_file).replace('hyp_', '')
        model_names.append(model_name)

print(f"model_names: {model_names}")

# Verify length 
num_lines = len(src_lines)
for i, tlines in enumerate(tgt_lines):
    if len(tlines) != num_lines:
        print(f"Error: Translation file {model_names[i]} has different number of lines than source.")
        sys.exit(1)


# Process each tgt file
print("Processing translations comparatively...")

js_data = []
for line_idx in range(num_lines):
    translations = [tgt_lines[i][line_idx] for i in range(len(tgt_lines))]

    js_data.append({
        'id': line_idx,
        'source': src_lines[line_idx],
        'translations': translations,
        'model_names': model_names
    })

# Proces with threading
print("Processing with threading...")

reviews = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for js in js_data:
        future = executor.submit(request_gpt4, js)
        futures.append(future)
    
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Processing comparative rankings"):
        reviews.append(future.result())

reviews.sort(key=lambda x: x['id']) # sort by id


# Outputs
# output_dir = os.path.join(f"tbt-cn-200/gpt-4_mev_scores/comparative/5/")
output_dir = os.path.join(f"cn-tbt-200/gpt-4_mev_scores/4/")

os.makedirs(output_dir, exist_ok=True)
outfile = os.path.join(output_dir, "gpt-4_detailed_scores.json")

# Detailed score results (everything)
with open(outfile, 'w', encoding='utf-8') as fout:
    for review in reviews:
        json.dump(review, fout, ensure_ascii=False)
        fout.write('\n')

# individual score files (only scores)
score_files = []
for i, model_name in enumerate(model_names):
    score_file = os.path.join(output_dir, f"gpt-4_{model_name}")
    score_files.append(score_file)

    with open(score_file, 'w', encoding='utf-8') as fout:
        for review in reviews:
            score = review.get('scores', [0, 0, 0])[i]
            fout.write(f"{score}\n")

print(f"Results saved to: {outfile}.")
print(f"Individual scores saved to: {score_files}")
