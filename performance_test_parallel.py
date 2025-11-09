import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from preprocess.align import load_and_align
from neuralhash.adapter import compute_hash_bits
from neuralhash.db import load_db as load_nh_db
from hdic.db import load_db as load_hdic_db
from hdic.adapter import encode_hv
from fusion.parallel import best_neuralhash_distance, best_hdic_distance
from common.hamming import hamming_distance_bits


def normalize_similarity_from_hamming(d, n_bits):
    """Convert Hamming distance to similarity in [0,1]."""
    if d is None:
        return 0.0
    s = 1.0 - (float(d) / float(n_bits))
    return max(0.0, min(1.0, s))


def compute_sfinal(nh_dist, hdic_dist, w_nh):
    """Compute final similarity score using weighted fusion."""
    w_hdic = 1.0 - w_nh
    nh_sim = normalize_similarity_from_hamming(nh_dist, 96)
    hdic_sim = normalize_similarity_from_hamming(hdic_dist, 10000)
    sfinal = w_nh * nh_sim + w_hdic * hdic_sim
    return sfinal


def get_person_images(dataset_root):
    """Get all images organized by person ID."""
    person_images = {}
    dataset_path = Path(dataset_root)
    
    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue
        
        person_id = person_dir.name
        images = []
        
        for img_file in sorted(person_dir.iterdir()):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                images.append(str(img_file))
        
        if images:
            person_images[person_id] = images
    
    return person_images


def perform_testing(dataset_root, nh_db, hdic_db, selected_person, w_nh, Tnh, Thdic):
    """
    Perform genuine and impostor matching for a selected person.
    
    Genuine: Compare selected person's images with their own DB entries
    Impostor: Compare other persons' first image with selected person's DB entries
    """
    person_images = get_person_images(dataset_root)
    
    if selected_person not in person_images:
        raise ValueError(f"Selected person {selected_person} not found in dataset")
    
    genuine_scores = []
    impostor_scores = []
    
    # Get selected person's NH and HDIC from database
    selected_nh = None
    selected_hdic = None
    
    for entry in nh_db:
        if entry.get('person_id') == selected_person:
            selected_nh = entry
            break
    
    for entry in hdic_db:
        if entry.get('person_id') == selected_person:
            selected_hdic = entry
            break
    
    if not selected_nh or not selected_hdic:
        raise ValueError(f"Selected person {selected_person} not found in databases")
    
    print(f"\n[INFO] Testing with person: {selected_person}")
    print(f"[INFO] Weight w_nh={w_nh:.2f}, w_hdic={1-w_nh:.2f}")
    
    # Extract HDIC prototypes (handle dictionary structure)
    hdic_prototypes = []
    if isinstance(selected_hdic.get('prototypes'), dict):
        # prototypes is a dictionary like {"cluster_0": [...], "cluster_1": [...]}
        for cluster_name, proto_list in selected_hdic['prototypes'].items():
            hdic_prototypes.append(proto_list)
    else:
        # prototypes is already a list
        hdic_prototypes = selected_hdic.get('prototypes', [])
    
    # Genuine matches: Compare selected person's images with their own DB entries
    print("\n[INFO] Computing genuine matches...")
    for img_path in tqdm(person_images[selected_person], desc="Genuine"):
        probe = load_and_align(img_path)
        if probe is None:
            continue
        
        probe_bits = compute_hash_bits(probe)
        probe_hv = encode_hv(probe)
        
        # Compute NH distance
        nh_dists = []
        for db_hash in selected_nh['hashes']:
            db_bits = np.array(db_hash, dtype=np.uint8)
            dist = hamming_distance_bits(probe_bits, db_bits)
            nh_dists.append(dist)
        min_nh_dist = min(nh_dists) if nh_dists else None
        
        # Compute HDIC distance
        hdic_dists = []
        for db_hv in hdic_prototypes:
            db_hv_bits = np.array(db_hv, dtype=np.uint8)
            dist = hamming_distance_bits(probe_hv, db_hv_bits)
            hdic_dists.append(dist)
        min_hdic_dist = min(hdic_dists) if hdic_dists else None
        
        # Compute final similarity score
        if min_nh_dist is not None and min_hdic_dist is not None:
            sfinal = compute_sfinal(min_nh_dist, min_hdic_dist, w_nh)
            genuine_scores.append(sfinal)
    
    # Impostor matches: Compare other persons' first image with selected person's DB
    print("\n[INFO] Computing impostor matches...")
    for person_id, images in tqdm(person_images.items(), desc="Impostor"):
        if person_id == selected_person:
            continue
        
        if len(images) == 0:
            continue
        
        # Use first image of each impostor
        img_path = images[0]
        probe = load_and_align(img_path)
        if probe is None:
            continue
        
        probe_bits = compute_hash_bits(probe)
        probe_hv = encode_hv(probe)
        
        # Compute NH distance against selected person's DB
        nh_dists = []
        for db_hash in selected_nh['hashes']:
            db_bits = np.array(db_hash, dtype=np.uint8)
            dist = hamming_distance_bits(probe_bits, db_bits)
            nh_dists.append(dist)
        min_nh_dist = min(nh_dists) if nh_dists else None
        
        # Compute HDIC distance against selected person's DB
        hdic_dists = []
        for db_hv in hdic_prototypes:
            db_hv_bits = np.array(db_hv, dtype=np.uint8)
            dist = hamming_distance_bits(probe_hv, db_hv_bits)
            hdic_dists.append(dist)
        min_hdic_dist = min(hdic_dists) if hdic_dists else None
        
        # Compute final similarity score
        if min_nh_dist is not None and min_hdic_dist is not None:
            sfinal = compute_sfinal(min_nh_dist, min_hdic_dist, w_nh)
            impostor_scores.append(sfinal)
    
    return np.array(genuine_scores), np.array(impostor_scores)


def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    """Compute FAR and FRR for different thresholds."""
    far_list = []
    frr_list = []
    
    for thresh in thresholds:
        # FAR: False Accept Rate - impostors accepted
        fa = np.sum(impostor_scores >= thresh)
        far = fa / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        
        # FRR: False Reject Rate - genuine rejected
        fr = np.sum(genuine_scores < thresh)
        frr = fr / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        
        far_list.append(far)
        frr_list.append(frr)
    
    return np.array(far_list), np.array(frr_list)


def compute_accuracy(genuine_scores, impostor_scores, threshold):
    """Compute accuracy at a given threshold."""
    # True Positives: genuine accepted
    tp = np.sum(genuine_scores >= threshold)
    # True Negatives: impostors rejected
    tn = np.sum(impostor_scores < threshold)
    
    total = len(genuine_scores) + len(impostor_scores)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    
    return accuracy


def compute_metrics_at_threshold(genuine_scores, impostor_scores, threshold):
    """Compute detailed metrics at a specific threshold."""
    tp = np.sum(genuine_scores >= threshold)
    fn = np.sum(genuine_scores < threshold)
    tn = np.sum(impostor_scores < threshold)
    fp = np.sum(impostor_scores >= threshold)
    
    total = len(genuine_scores) + len(impostor_scores)
    
    # Basic metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as TPR
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Rates
    tpr = tp / len(genuine_scores) if len(genuine_scores) > 0 else 0.0  # True Positive Rate
    fpr = fp / len(impostor_scores) if len(impostor_scores) > 0 else 0.0  # False Positive Rate
    tnr = tn / len(impostor_scores) if len(impostor_scores) > 0 else 0.0  # True Negative Rate
    fnr = fn / len(genuine_scores) if len(genuine_scores) > 0 else 0.0  # False Negative Rate
    
    return {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'far': fpr,  # FAR = FPR
        'frr': fnr   # FRR = FNR
    }


def main():
    parser = argparse.ArgumentParser(description="Performance test for parallel fusion technique")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--selected_person", required=True, help="Person ID to use for testing")
    parser.add_argument("--Tnh", type=int, default=30, help="NeuralHash threshold")
    parser.add_argument("--Thdic", type=int, default=3100, help="HDIC threshold")
    parser.add_argument("--output_dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load databases
    print("[INFO] Loading databases...")
    nh_db = load_nh_db()
    hdic_db = load_hdic_db()
    
    print(f"[INFO] Loaded {len(nh_db)} NeuralHash entries")
    print(f"[INFO] Loaded {len(hdic_db)} HDIC entries")
    
    # Weight values to test
    weight_values = np.arange(0.0, 1.05, 0.1)
    
    results = []
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT WEIGHT COMBINATIONS")
    print("="*60)
    
    for w_nh in weight_values:
        w_nh = round(w_nh, 2)
        w_hdic = round(1.0 - w_nh, 2)
        
        print(f"\n--- Testing w_nh={w_nh}, w_hdic={w_hdic} ---")
        
        # Perform testing
        genuine_scores, impostor_scores = perform_testing(
            args.dataset, nh_db, hdic_db, args.selected_person, w_nh, args.Tnh, args.Thdic
        )
        
        print(f"[RESULTS] Genuine scores: {len(genuine_scores)}, mean={np.mean(genuine_scores):.4f}, std={np.std(genuine_scores):.4f}")
        print(f"[RESULTS] Impostor scores: {len(impostor_scores)}, mean={np.mean(impostor_scores):.4f}, std={np.std(impostor_scores):.4f}")
        
        # Compute FAR/FRR for different thresholds
        thresholds = np.linspace(0, 1, 100)
        far, frr = compute_far_frr(genuine_scores, impostor_scores, thresholds)
        
        # Find EER (Equal Error Rate)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # Compute detailed metrics at EER threshold
        eer_metrics = compute_metrics_at_threshold(genuine_scores, impostor_scores, eer_threshold)
        
        # Compute accuracy at EER threshold
        accuracy_at_eer = eer_metrics['accuracy']
        
        results.append({
            'w_nh': w_nh,
            'w_hdic': w_hdic,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'accuracy_at_eer': accuracy_at_eer,
            'eer_metrics': eer_metrics,
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores,
            'thresholds': thresholds,
            'far': far,
            'frr': frr,
            'genuine_mean': np.mean(genuine_scores),
            'genuine_std': np.std(genuine_scores),
            'impostor_mean': np.mean(impostor_scores),
            'impostor_std': np.std(impostor_scores)
        })
        
        print(f"[RESULTS] EER={eer:.4f} at threshold={eer_threshold:.4f}")
        print(f"[RESULTS] Accuracy at EER={accuracy_at_eer:.4f}")
        print(f"[RESULTS] Precision={eer_metrics['precision']:.4f}, Recall={eer_metrics['recall']:.4f}, F1={eer_metrics['f1_score']:.4f}")
    
    # Plot FAR vs FRR for all weights
    print("\n[INFO] Generating FAR vs FRR plots...")
    plt.figure(figsize=(14, 8))
    
    for res in results:
        label = f"w_nh={res['w_nh']:.1f}"
        plt.plot(res['far'], res['frr'], marker='o', markersize=3, label=label, alpha=0.7)
    
    # Add diagonal line (EER line)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='EER line')
    
    plt.xlabel('False Accept Rate (FAR)', fontsize=12)
    plt.ylabel('False Reject Rate (FRR)', fontsize=12)
    plt.title(f'FAR vs FRR for Different Weights\n(Person: {args.selected_person}, Tnh={args.Tnh}, Thdic={args.Thdic})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'far_vs_frr.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir / 'far_vs_frr.png'}")
    plt.close()
    
    # Plot FAR and FRR vs Weight
    print("[INFO] Generating FAR and FRR vs Weight plots...")
    plt.figure(figsize=(12, 6))
    
    weights = [res['w_nh'] for res in results]
    fars_at_eer = [res['eer_metrics']['far'] for res in results]
    frrs_at_eer = [res['eer_metrics']['frr'] for res in results]
    eers = [res['eer'] for res in results]
    
    plt.plot(weights, fars_at_eer, marker='o', linewidth=2, markersize=8, label='FAR at EER', color='red')
    plt.plot(weights, frrs_at_eer, marker='s', linewidth=2, markersize=8, label='FRR at EER', color='blue')
    plt.plot(weights, eers, marker='^', linewidth=2, markersize=8, label='EER', color='green', linestyle='--')
    
    plt.xlabel('w_nh (NeuralHash Weight)', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title('FAR & FRR vs Weight at EER Point', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, max(max(fars_at_eer), max(frrs_at_eer)) * 1.1])
    
    # Mark best EER
    best_eer_idx = np.argmin(eers)
    best_eer_w_nh = weights[best_eer_idx]
    best_eer = eers[best_eer_idx]
    plt.axvline(best_eer_w_nh, color='purple', linestyle='--', alpha=0.5, linewidth=2)
    plt.text(best_eer_w_nh, max(eers) * 0.8, f'Best EER\nw_nh={best_eer_w_nh:.1f}\nEER={best_eer:.4f}', 
             fontsize=9, color='purple', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'far_frr_vs_weight.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir / 'far_frr_vs_weight.png'}")
    plt.close()
    
    # Plot accuracy and other metrics vs weight
    print("[INFO] Generating accuracy and metrics plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    accuracies = [res['accuracy_at_eer'] for res in results]
    precisions = [res['eer_metrics']['precision'] for res in results]
    recalls = [res['eer_metrics']['recall'] for res in results]
    f1_scores = [res['eer_metrics']['f1_score'] for res in results]
    
    # Accuracy
    axes[0, 0].plot(weights, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_xlabel('w_nh (NeuralHash Weight)', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy at EER', fontsize=11)
    axes[0, 0].set_title('Accuracy at EER vs Weight', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    best_acc_idx = np.argmax(accuracies)
    best_acc_w_nh = weights[best_acc_idx]
    best_accuracy = accuracies[best_acc_idx]
    axes[0, 0].axvline(best_acc_w_nh, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].text(best_acc_w_nh, best_accuracy, f'  Best: {best_accuracy:.4f}\n  w_nh={best_acc_w_nh:.1f}', 
                    fontsize=9, color='red')
    
    # EER
    axes[0, 1].plot(weights, eers, marker='s', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('w_nh (NeuralHash Weight)', fontsize=11)
    axes[0, 1].set_ylabel('Equal Error Rate (EER)', fontsize=11)
    axes[0, 1].set_title('EER vs Weight', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, max(eers) * 1.1])
    
    axes[0, 1].axvline(best_eer_w_nh, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].text(best_eer_w_nh, best_eer, f'  Best: {best_eer:.4f}\n  w_nh={best_eer_w_nh:.1f}', 
                    fontsize=9, color='red')
    
    # Precision and Recall
    axes[1, 0].plot(weights, precisions, marker='^', linewidth=2, markersize=8, label='Precision', color='orange')
    axes[1, 0].plot(weights, recalls, marker='v', linewidth=2, markersize=8, label='Recall', color='purple')
    axes[1, 0].set_xlabel('w_nh (NeuralHash Weight)', fontsize=11)
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_title('Precision & Recall at EER vs Weight', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # F1 Score
    axes[1, 1].plot(weights, f1_scores, marker='D', linewidth=2, markersize=8, color='darkred')
    axes[1, 1].set_xlabel('w_nh (NeuralHash Weight)', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title('F1 Score at EER vs Weight', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1_w_nh = weights[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    axes[1, 1].axvline(best_f1_w_nh, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].text(best_f1_w_nh, best_f1, f'  Best: {best_f1:.4f}\n  w_nh={best_f1_w_nh:.1f}', 
                    fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_vs_weight.png', dpi=300, bbox_inches='tight')
    print(f"[SAVED] {output_dir / 'metrics_vs_weight.png'}")
    plt.close()
    
    # Save detailed results to JSON
    results_summary = {
        'selected_person': args.selected_person,
        'Tnh': args.Tnh,
        'Thdic': args.Thdic,
        'weights_tested': [
            {
                'w_nh': res['w_nh'],
                'w_hdic': res['w_hdic'],
                'eer': float(res['eer']),
                'eer_threshold': float(res['eer_threshold']),
                'accuracy_at_eer': float(res['accuracy_at_eer']),
                'num_genuine': len(res['genuine_scores']),
                'num_impostor': len(res['impostor_scores']),
                'genuine_mean': float(res['genuine_mean']),
                'genuine_std': float(res['genuine_std']),
                'impostor_mean': float(res['impostor_mean']),
                'impostor_std': float(res['impostor_std']),
                'metrics_at_eer': {
                    'tp': res['eer_metrics']['tp'],
                    'tn': res['eer_metrics']['tn'],
                    'fp': res['eer_metrics']['fp'],
                    'fn': res['eer_metrics']['fn'],
                    'accuracy': float(res['eer_metrics']['accuracy']),
                    'precision': float(res['eer_metrics']['precision']),
                    'recall': float(res['eer_metrics']['recall']),
                    'f1_score': float(res['eer_metrics']['f1_score']),
                    'far': float(res['eer_metrics']['far']),
                    'frr': float(res['eer_metrics']['frr']),
                    'tpr': float(res['eer_metrics']['tpr']),
                    'fpr': float(res['eer_metrics']['fpr'])
                }
            }
            for res in results
        ]
    }
    
    results_file = output_dir / 'performance_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"[SAVED] {results_file}")
    
    # Create a detailed CSV report
    csv_file = output_dir / 'performance_results.csv'
    with open(csv_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['w_nh', 'w_hdic', 'EER', 'EER_Threshold', 'Accuracy_at_EER', 
                        'Precision', 'Recall', 'F1_Score', 'FAR', 'FRR',
                        'TP', 'TN', 'FP', 'FN', 
                        'Genuine_Mean', 'Genuine_Std', 'Impostor_Mean', 'Impostor_Std'])
        
        for res in results:
            writer.writerow([
                res['w_nh'], res['w_hdic'], res['eer'], res['eer_threshold'], res['accuracy_at_eer'],
                res['eer_metrics']['precision'], res['eer_metrics']['recall'], res['eer_metrics']['f1_score'],
                res['eer_metrics']['far'], res['eer_metrics']['frr'],
                res['eer_metrics']['tp'], res['eer_metrics']['tn'], 
                res['eer_metrics']['fp'], res['eer_metrics']['fn'],
                res['genuine_mean'], res['genuine_std'], res['impostor_mean'], res['impostor_std']
            ])
    print(f"[SAVED] {csv_file}")
    
    # Print detailed summary table
    print("\n" + "="*100)
    print("DETAILED RESULTS TABLE")
    print("="*100)
    print(f"{'w_nh':<6} {'w_hdic':<8} {'EER':<8} {'Acc@EER':<10} {'Prec':<8} {'Rec':<8} {'F1':<8} {'FAR':<8} {'FRR':<8}")
    print("-"*100)
    for res in results:
        print(f"{res['w_nh']:<6.2f} {res['w_hdic']:<8.2f} {res['eer']:<8.4f} {res['accuracy_at_eer']:<10.4f} "
              f"{res['eer_metrics']['precision']:<8.4f} {res['eer_metrics']['recall']:<8.4f} "
              f"{res['eer_metrics']['f1_score']:<8.4f} {res['eer_metrics']['far']:<8.4f} "
              f"{res['eer_metrics']['frr']:<8.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest Accuracy at EER: {best_accuracy:.4f} at w_nh={best_acc_w_nh:.2f}, w_hdic={1-best_acc_w_nh:.2f}")
    print(f"Best EER: {best_eer:.4f} at w_nh={best_eer_w_nh:.2f}, w_hdic={1-best_eer_w_nh:.2f}")
    print(f"Best F1 Score: {best_f1:.4f} at w_nh={best_f1_w_nh:.2f}, w_hdic={1-best_f1_w_nh:.2f}")
    
    best_result = results[best_eer_idx]
    print(f"\nDetailed metrics at best EER (w_nh={best_eer_w_nh:.2f}):")
    print(f"  - Accuracy: {best_result['accuracy_at_eer']:.4f}")
    print(f"  - Precision: {best_result['eer_metrics']['precision']:.4f}")
    print(f"  - Recall: {best_result['eer_metrics']['recall']:.4f}")
    print(f"  - F1 Score: {best_result['eer_metrics']['f1_score']:.4f}")
    print(f"  - FAR: {best_result['eer_metrics']['far']:.4f}")
    print(f"  - FRR: {best_result['eer_metrics']['frr']:.4f}")
    print(f"  - TP: {best_result['eer_metrics']['tp']}, TN: {best_result['eer_metrics']['tn']}")
    print(f"  - FP: {best_result['eer_metrics']['fp']}, FN: {best_result['eer_metrics']['fn']}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Plots: far_vs_frr.png, far_frr_vs_weight.png, metrics_vs_weight.png")
    print(f"  - Data: performance_results.json, performance_results.csv")


if __name__ == "__main__":
    main()