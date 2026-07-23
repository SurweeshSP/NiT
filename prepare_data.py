import json
import random
from collections import Counter
import re
import os
import pandas as pd

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'xxxx+', 'xxxx', text)
    text = re.sub(r'\{\$\d+\.\d+\}', '{$[OTP].00}', text)
    # Basic noise removal
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("Loading data1.json...")
    try:
        with open('data/data1.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Total raw records: {len(data)}")
    
    report_lines = [
        "# Dataset Quality Audit Report",
        "",
        f"**Initial Raw Records:** {len(data)}",
        ""
    ]
    
    # 1. Remove empty and extremely short complaints (Noise)
    valid_data = []
    noisy_count = 0
    for d in data:
        text = d.get('complaint_what_happened', '').strip()
        if len(text) < 20: # Arbitrary threshold for noisy/short
            noisy_count += 1
        else:
            valid_data.append(d)
            
    report_lines.append(f"**Noisy/Empty Samples Removed:** {noisy_count}")
    
    # 2. Duplicate Detection
    seen_texts = set()
    unique_data = []
    duplicate_count = 0
    for d in valid_data:
        text = clean_text(d['complaint_what_happened'])
        if text in seen_texts:
            duplicate_count += 1
        else:
            seen_texts.add(text)
            d['complaint_what_happened_clean'] = text
            unique_data.append(d)
            
    report_lines.append(f"**Duplicate Complaints Removed:** {duplicate_count}")
    report_lines.append(f"**Records after basic cleaning:** {len(unique_data)}")
    report_lines.append("")
    
    # 3. Label Distribution and Top 5 selection
    product_counts = Counter(d['product'] for d in unique_data if 'product' in d)
    top_5 = [item[0] for item in product_counts.most_common(5)]
    
    report_lines.append("### Initial Class Distribution (Top 5 Classes)")
    for label, count in product_counts.most_common(5):
        report_lines.append(f"- **{label}**: {count}")
    report_lines.append("")
    
    # Filter to top 5
    filtered_data = [d for d in unique_data if d.get('product') in top_5]
    for d in filtered_data:
        d['label'] = d['product']
        
    # 4. Class Balancing (Downsample to median to prevent extreme skew)
    class_groups = {label: [] for label in top_5}
    for d in filtered_data:
        class_groups[d['label']].append(d)
        
    counts = [len(v) for v in class_groups.values()]
    median_count = int(sorted(counts)[len(counts)//2])
    
    report_lines.append(f"**Balancing Strategy:** Downsample majority classes to median count ({median_count}). Minority classes are kept as is.")
    
    balanced_data = []
    for label, items in class_groups.items():
        if len(items) > median_count:
            random.seed(42)
            sampled = random.sample(items, median_count)
            balanced_data.extend(sampled)
        else:
            balanced_data.extend(items)
            
    # Final counts
    final_counts = Counter(d['label'] for d in balanced_data)
    report_lines.append("")
    report_lines.append("### Final Balanced Class Distribution")
    for label, count in final_counts.most_common():
        report_lines.append(f"- **{label}**: {count}")
        
    report_lines.append("")
    report_lines.append(f"**Final Trainable Records:** {len(balanced_data)}")
    
    label_mapping = {label: i for i, label in enumerate(top_5)}
    
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(balanced_data)
    split_idx = int(len(balanced_data) * 0.8)
    
    train_data = balanced_data[:split_idx]
    test_data = balanced_data[split_idx:]
    
    report_lines.append(f"**Train Split:** {len(train_data)}")
    report_lines.append(f"**Test Split:** {len(test_data)}")
    
    # Write files
    os.makedirs('results', exist_ok=True)
    with open('results/label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=4)
        
    with open('data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open('data/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
        
    with open('dataset_quality_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print("Data preparation complete! Dataset quality report generated.")

if __name__ == '__main__':
    main()
