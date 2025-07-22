import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict


# MODELS = ['deepseek-v3', 'google-translate', 'qwen2.5_72b', 'qwen3_32b']
MODELS = ['modelA', 'modelB', 'modelC'] #, 'modelD'

# def rank_models_per_line(model_scores, lineA, lineB, lineC, lineD):
#     """Rank models for each line based on chrF scores."""
    
#     rankings = []
#     for i in range(num_lines):
#         # Get scores for this line across all models
#         line_scores = {model: model_scores[model][i] for model in model_names}
        
#         # Sort models by score (descending)
#         ranked = sorted(line_scores.items(), key=lambda x: -x[1])
        
#         rankings.append(ranked)
    
#     return rankings


def load_scores(filename) -> list:
    """Load scores from file into list."""
    scores = []
    with open(filename, 'r') as f:
        for idx, line in enumerate(f, 1):
            scores.append(float(line.strip()))
    return scores


# def compare_models_pairwise(scores_dict, idx):
#     """Compare all model pairs for a line idx."""
#     comparisons = {}

#     for i in range(len(MODELS)):
#         for j in range(i + 1, len(MODELS)):
#             model_a, model_b = MODELS[i], MODELS[j]
#             score_a = scores_dict[model_a]
#             score_b = scores_dict[model_b]

#             if score_a is None or score_b is None:
#                 comparisons[f"{model_a}_vs_{model_b}"] = "invalid"
#             elif abs(score_a - score_b) < 1e-6: # EQUAL
#                 comparisons[f"{model_a}_vs_{model_b}"] = "tie"
#             elif score_a > score_b:
#                 comparisons[f"{model_a}_vs_{model_b}"] = f"{model_a}"
#             else:
#                 comparisons[f"{model_a}_vs_{model_b}"] = f"{model_b}"
            
#     return comparisons


def analyze_agreement_with_human(model_scores, human_scores):
    """Analyze how well model scores correlate with human annotations."""
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(model_scores, human_scores)
    spearman_corr, spearman_p = spearmanr(model_scores, human_scores)
    kendall_corr, kendall_p = kendalltau(model_scores, human_scores)

    valid_pairs = [(m, h) for m, h in zip(model_scores, human_scores)]
    
    # Calculate error metrics
    # mse = np.mean((m - h)**2 for m, h in valid_pairs)
    # mae = np.mean(abs(m - h) for m, h in valid_pairs)
    mae = np.mean([abs(m - h) for m, h in valid_pairs])
    mse = np.mean([(m - h) ** 2 for m, h in valid_pairs])

    # Agreement catagories 
    exact_match = sum(1 for m, h in valid_pairs if abs(m - h) < 0.1)
    close_match = sum(1 for m, h in valid_pairs if abs(m - h) <= 1.0)
    major_disagreement = sum(1 for m, h in valid_pairs if abs(m - h) >= 2)

    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr, 
        "kendall_corr": kendall_corr,
        "pearson_p": pearson_p,
        "spearman_p": spearman_p,
        "kendall_p": kendall_p,
        "mse": mse,
        "mae": mae,
        "valid_samples": len(valid_pairs),
        "exact_match": exact_match,
        "close_match": close_match,
        "major_disagreement": major_disagreement
    }


def identify_fault_patterns(model_scores, human_scores, model_name, threshold=1.0):
    """Identify where and how the model fails compared to human annotation."""
    faults = []

    for i, (model_score, human_score) in enumerate(zip(model_scores, human_scores), 1):
        if model_score is None or human_score is None:
            continue

        error = abs(model_score - human_score)
        if error >= threshold:
            fault_type = "overestimate" if model_score > human_score else "underestimate"
            faults.append({
                "line": i,
                "model_score": model_score,
                "human_score": human_score,
                "error": error,
                "fault_type": fault_type
            })

    return faults


def line_by_line_agreement(mev_scores, hev_scores, num_lines):
    """3. LINE-BY-LINE AGREEMENT"""
    agreement_count = {
        'Good': {
            'equivelant': 0,
            'overestimate': 0,
            'underestimate': 0
        },
        'Fair': {
            'overestimate': 0,
            'underestimate': 0
        },
        'Poor': {
            'overestimate': 0,
            'underestimate': 0
        }
    }


    print("=== 3.1, LINE-BY-LINE AGREEMENT ===")
    for idx in range(num_lines):
        print(f"\nLine {idx + 1}:")
    
        # Calculate correlation for this line across all models
        line_model_scores = []
        line_human_rankings = []
    
        for model in MODELS:
            mev_score = mev_scores[model][idx]
            hev_score = hev_scores[model][idx]

            # Print error
            error = abs(mev_score - hev_score)
            agreement = "Good" if error <= 0.5 else "Fair" if error <= 1.0 else f"Poor"
            if error >= 1.0:
                fault_type = "overestimate" if mev_score > hev_score else "underestimate"
                agreement_count[agreement][fault_type] += 1                
            else:
                fault_type = "equivelant"
                agreement_count['Good']['equivelant'] += 1
            print(f"  {model}: Model={mev_score:.2f}, Human={hev_score:.2f}, Error={error:.2f} ({agreement}, {fault_type})")
    print( # agreement summary
        f"\n\n=== 3.2, AGREEMENT COUNT SUMMARY ===\n"
        f"\nagreement_count: Good={sum(agreement_count['Good'].values())}, Fair={sum(agreement_count['Fair'].values())}, Poor={sum(agreement_count['Poor'].values())}\n"
        f"fault_type_count: equivelances={agreement_count['Good']['equivelant']} overestimates={sum(v['overestimate'] for v in agreement_count.values())}, underestimates={sum(v['underestimate'] for v in agreement_count.values())}\n"
    )
    print( # agreement detailed summary 
        f"Good_overestimate={agreement_count['Good']['overestimate']}\n"
        f"Good_underestimate={agreement_count['Good']['underestimate']}\n"
        f"Fair_overestimate={agreement_count['Fair']['overestimate']}\n"
        f"Fair_underestimate={agreement_count['Fair']['underestimate']}\n"
        f"Poor_overestimate={agreement_count['Poor']['overestimate']}\n"
        f"Poor_underestimate={agreement_count['Poor']['underestimate']}"
    )
        

def model_by_model_agreement(mev_scores, hev_scores):
    """2. MODEL BY MODEL AGREEMENT"""
    for model in MODELS:
        # Filter valid pairs
        valid_pairs = [(m, h) for m, h in zip(mev_scores[model], hev_scores[model]) 
                    if m is not None and h is not None]
    
        if len(valid_pairs) < 2:
            print(f"{model}: Insufficient valid data")
            continue
    
        model_vals, human_vals = zip(*valid_pairs)
    
        # Calculate correlations and errors
        results = analyze_agreement_with_human(model_vals, human_vals)

        print(f"{model}:")
        print(f"  Pearson correlation: {results['pearson_corr']:.4f}")
        # print(f"  Spearman correlation: {results['spearman_corr']:.4f}")
        print(f"  Kendalltau correlation: {results['kendall_corr']:.4f}")
        print(f"  Mean Absolute Error: {results['mae']:.4f}")
        print(f"  Mean Squared Error: {results['mse']:.4f}\n")
        print(f"  Exact matches: {results['exact_match']}/{results['valid_samples']}")
        print(f"  Close matches: {results['close_match']}/{results['valid_samples']}")
        print(f"  Major disagreements: {results['major_disagreement']}/{results['valid_samples']}\n")

    print("\n")


def agreement(mev_scores, hev_scores):
    """AGREEMENT"""
    valid_pairs = []
    all_mev_scores = []
    all_hev_scores = []

    for model in MODELS:
        # Filter valid pairs
        all_mev_scores += mev_scores[model]
        all_hev_scores += hev_scores[model]

    valid_pairs = [(m, h) for m, h in zip(all_mev_scores, all_hev_scores) 
                if m is not None and h is not None]
    
    if len(valid_pairs) < 2:
        print("Insufficient valid data")
    
    model_vals, human_vals = zip(*valid_pairs)
    
    # Calculate correlations and errors
    results = analyze_agreement_with_human(model_vals, human_vals)

    print(f"  Pearson correlation: {results['pearson_corr']:.4f}")
    # print(f"  Spearman correlation: {results['spearman_corr']:.4f}")
    print(f"  Kendalltau correlation: {results['kendall_corr']:.4f}")
    print(f"  Mean Absolute Error: {results['mae']:.4f}")
    print(f"  Mean Squared Error: {results['mse']:.4f}\n")
    print(f"  Exact matches: {results['exact_match']}/{results['valid_samples']}")
    print(f"  Close matches: {results['close_match']}/{results['valid_samples']}")
    print(f"  Major disagreements: {results['major_disagreement']}/{results['valid_samples']}\n")


def main():
    """
    Usage: python3 scripts/rank_models.py
    """
    # if len(sys.argv) != 6:
    #     print("Usage: python rank_models.py <modelA_file> <modelB_file> <modelC_file> <modelD_file> <human_file>")
    #     print("Node: Assumes 5th file contains human annotations (1-5 scale)")
    
    # metric = "chrF++"
    # metric = "BERTScore"
    # metric = "NR_BERTScore"
    metric = "gpt-4"

    # sent_scores_path = f"tbt-cn-200/{metric}_mev_scores/"
    # sent_scores_path = f"tbt-cn-200/{metric}_mev_scores/comparative/0/"
    sent_scores_path = f"cn-tbt-200/{metric}_mev_scores/1/"

    mevA = sent_scores_path + f"{metric}_deepseek-v3"
    mevB = sent_scores_path + f"{metric}_google-translate"
    mevC = sent_scores_path + f"{metric}_qwen2.5_72b"
    # mevD = sent_scores_path + f"{metric}_qwen3_32b"

    # hev_annotated_path = "tbt-cn-200/hev_scores/"
    hev_annotated_path = "cn-tbt-200/hev_scores/"
    hevA = hev_annotated_path + "hev_deepseek-v3"
    hevB = hev_annotated_path + "hev_google-translate"
    hevC = hev_annotated_path + "hev_qwen2.5_72b"
    # hevD = hev_annotated_path + "hev_qwen3_32b"
    # fileA, fileB, fileC, fileD, human_file = sys.argv[1:]

    mev_scores = { # machine evaluated scores
        MODELS[0]: load_scores(mevA),
        MODELS[1]: load_scores(mevB),
        MODELS[2]: load_scores(mevC)
        # MODELS[3]: load_scores(mevD)
    }
    hev_scores = {
        MODELS[0]: load_scores(hevA),
        MODELS[1]: load_scores(hevB),
        MODELS[2]: load_scores(hevC)
        # MODELS[3]: load_scores(hevD)
    }
    # human_scores = load_scores(human_file)

    # verify files are same lengths 
    lengths = [len(mev_scores[model]) for model in mev_scores] + [len(hev_scores[model]) for model in hev_scores]
    if len(set(lengths)) > 1:
        print(f"Error: Files have different lengths: {dict(zip(list(mev_scores.keys()) + list(hev_scores.keys()), lengths))}")
        sys.exit(1)

    num_lines = lengths[0]
    print(f"Analyzing {num_lines} lines of data.")

    # Analyze by line and by model 
    print("\n=== 1. OVERALL AGREEMENT ===\n")
    agreement(mev_scores, hev_scores)

    print("\n=== 2. MODEL-BY-MODEL AGREEMENT ===\n")
    model_by_model_agreement(mev_scores, hev_scores)

    line_by_line_agreement(mev_scores, hev_scores, lengths[0])

 
if __name__ == "__main__":
    main()