import requests
import json
import os
import sys
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

api_key = "youdao-ai-IfdzwyFarZiDUnOx"


def get_prompt(tibetan_source, chinese_translation):
    prompt = f"""You will rank the quality of a Chinese translation of Tibetan text.
            Source (Tibetan): {tibetan_source}
            Translation (Chinese): {chinese_translation}
            
            Rate this translation from 0.0000 to 1.0000 (4 decimal places) based on:
            
            - Accuracy of translation from Tibetan source
            - Fluency and naturalness of Chinese output
            - Preservation of meaning and context
            - Grammatical correctness
            
            Output only the numerical score (e.g., 0.8750). No additional text."""
    return prompt


def request_gpt4(js_data):
    time.sleep(0.1)

    tibetan_source = js_data['tibetan_source']
    chinese_translation = js_data['chinese_translation']
    prompt = get_prompt(tibetan_source, chinese_translation)
    messages=[{'role': 'user', 'content':  prompt}]
    
    data = {
        "api-key": api_key,
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0.0,
        "maxTokens": 50
    }

    for i in range(3):
        try:
            response = requests.post("http://yd-chatgpt-api.inner.youdao.com/api", json=data)
            result = response.json()
            # print("DEBUG - Raw API Response:", result)

            # print("SCORE :::::::::::::::::::::::::::::::::::::::::::\n")
            score = result['detail']['choices'][0]['message']['content'].strip()
            # print("====\n")
            # print(f"SCORE - {score}")

            # try:
            float_score = float(score)
            if float_score >= 0.0 and float_score <= 1.0:
                js_data['score'] = f"{float_score:.4f}"
            else:
                print(f'#RANGE_ERROR#--{response}')
                js_data['score'] = "#RANGE_ERROR#"
            # except ValueError:
            #     print(f'#VALUE_ERROR#--{response}')
            #     js_data['score'] = "#VALUE_ERROR#"
            return js_data
        except Exception as e:
            if i == 2:
                js_data['score'] = "#ERROR#"
                print(f'#ERROR#--{response}')
            else:
                print("Full Response:", response.json())
        
        # if 'score' not in js_data:
        #     js_data['score'] = "#MISSING_SCORE#"
    return js_data


# Expected arguments: tibetan_file translation1_file translation2_file translation3_file translation4_file output_file
# Usage: python3 scripts/flores-test-gpt-4.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b tbt-cn-200/mt-hyps/hyp_qwen3_32b

if len(sys.argv) != 6:
    print("Usage: python script.py <tibetan_file> <tgt1_file> <tgt2_file> <tgt3_file> <tgt4_file>")
    sys.exit(1)

src_file, tgt1_file, tgt2_file, tgt3_file, tgt4_file = sys.argv[1:]


with open(src_file, 'r', encoding='utf-8') as f:
    src_lines = [line.strip() for line in f.readlines()]

tgt_files = [tgt1_file, tgt2_file, tgt3_file, tgt4_file]
tgt_lines = []

for tgt_file in tgt_files:
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines.append([line.strip() for line in f.readlines()])


file_output_path = os.path.join(f"tbt-cn-200/gpt-4_mev_scores/")
file_outputs = []

for i in range(len(tgt_files)):
    file_outputs.append(f"{file_output_path}gpt4_ranking_{chr(ord('A')+ i)}")

        # Verify length 
num_lines = len(src_lines)
        # for i, tlines in enumerate(tgt_lines):
        #     if len(tlines) != num_lines:
        #         print(f"Error: Translation file {i+1} has different number of lines than source.")
        #         sys.exit(1)

# Process each tgt file

for file_idx, tlines in enumerate(tgt_lines):
    with open(file_outputs[file_idx], 'w', encoding='utf-8') as fout:
        print(f"Processing translation file {file_idx + 1}...")

        js_list = []
        for line_idx in range(num_lines):
            js_list.append({
                'id': line_idx,
                'tibetan_source': src_lines[line_idx],
                'chinese_translation': tlines[line_idx]
            })

        # Process with threading
        reviews = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for js in js_list:
                future = executor.submit(request_gpt4, js)
                futures.append(future)

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures), 
                desc=f"Translation file {file_idx + 1}"):
                reviews.append(future.result())

        reviews.sort(key=lambda x: x['id'])
        for review in reviews:
            score = review.get('score', -1)
            if score != -1:
                try:
                    score = f"{(float(score)*4 + 1):.4f}"
                except (ValueError, TypeError):
                    score = '#INVALID_SCORE#'
            else:
                score = '#INVALID_SCORE#'
            print(score.replace("\n","\\n"), file=fout)