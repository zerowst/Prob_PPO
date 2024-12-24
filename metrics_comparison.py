import pandas as pd
import matplotlib.pyplot as plt
import json


file_path = "/Users/xiongzheng/Downloads/result/compare/result.csv"  
data = pd.read_csv(file_path)


metrics = [
    "correct_frac", 
    "exact_match", 
    "exact_match_frac", 
    "majority_vote_acc", 
    "none_answer_extracted_frac_per_problem", 
    "once_hit", 
    "unique_answer_count"
]


for col in ["prob_test", "test", "dpo_test"]:
    data[col] = data[col].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
    for metric in metrics:
        data[f"{col}_{metric}"] = data[col].apply(lambda x: x.get(metric, None))


filtered_data = data[data["training_step"] <= 200]
titles = [
    "Correct_Acc", 
    "Exact Match", 
    "Exact Match Fraction", 
    "Majority_Voting_Acc", 
    "None Answer Extracted Fraction per Problem", 
    "Once Hit", 
    "Unique_Answer_Num"
]

labels = ["ProbPPO", "VinePPO", "DPO"]
font_settings = {'size': 20}  

for i, metric in enumerate(metrics):
    plt.figure(figsize=(12, 8))
    for j, col in enumerate(["prob_test", "test", "dpo_test"]):
        metric_col = f"{col}_{metric}"
        if metric_col in filtered_data:
            plt.plot(filtered_data["training_step"], filtered_data[metric_col], label=labels[j])
    

    plt.xlabel("Training Step", fontsize=font_settings['size'])
    plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=font_settings['size'])
    plt.title(f"{titles[i]}", fontsize=font_settings['size'])
    plt.legend(loc="best", fontsize=font_settings['size'] - 2)
    plt.grid(True)
    plt.tight_layout() 
    plt.savefig(f"figure/{metric.replace('_', ' ').capitalize()}.jpg", dpi=300) 
    plt.show()

