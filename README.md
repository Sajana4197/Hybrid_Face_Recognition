# Hybrid Face Recognition (NeuralHash + HDIC)

🚀 A hybrid face recognition system combining **Apple’s NeuralHash** (fast shortlist) and **Hyperdimensional Image Classification (HDIC)** (robust confirmation).  
Ideal for high-security applications like border control.

---

## 🔑 Features

- Unified preprocessing (MTCNN detect → align → crop → normalize)
- NeuralHash → 96-bit per-image hash signatures
- HDIC → 10,000D hypervector prototypes (multi-cluster per person)
- Two operation modes:
  - Cascade: NeuralHash shortlist → HDIC - confirmation (speed-biased)
- Parallel: NeuralHash + HDIC run in parallel and fuse scores (accuracy-biased)
- Open-set rejection via thresholds and fused decision
- JSONL databases for watchlist storage
- CLI tools: enroll, match, match_parallel, evaluate

---

## 📂 Structure

```bash
Hybrid_Face_Recognition/
├── cli/          # CLI tools (enroll, atch, evaluate, match_parallel)
├── common/       # Utilities (hamming, io)
├── preprocess/   # Face alignment (MTCNN)
├── neuralhash/   # NeuralHash pipeline (96-bit hashes)
├── hdic/         # HDIC pipeline (hypervectors)
├── fusion/       # Fusion strategies (cascade, parallel)
├── db/           # JSONL databases
├── dataset/      # Images (enroll, probe, test)

```

## ⚙️ Install

To install required packages run

```bash
pip install -r requirements.txt
```

## 📝 Usage

### 1. Enroll

To enroll a new person, run

```bash
python -m cli.enroll --id n000002 --name "John Doe" --images "dataset/n000002"
```

To enroll all persons at once from `dataset/enroll/`:

```bash
python -m cli.bulk_enroll --root dataset/enroll --clusters 3
```

### 2. Matching

#### i. Cascade mode (fast shortlist + confirmation)

```bash
python -m cli.match --image dataset/probe/unknown1.jpg --K 5
```

#### ii. Parallel mode (fused scores, accuracy-biased)

```bash
python -m cli.match_parallel --image dataset/probe/unknown1.jpg --Tnh 25 --Thdic 3000 --w_nh 0.4 --w_hdic 0.6 --fused_th 0.7 --require_both
```

### 3. Evaluate system

Closed-set:

```bash
python -m cli.evaluate_ident --test_root dataset/test --K 5 --Tnh 20,25,30 --Thdic 2000:4000:200
```

Open-set:

```bash
python -m cli.evaluate_ident --test_root dataset/test --unknown_root dataset/test_unknown --K 5 --Tnh 20,25,30 --Thdic 2000:4000:200
```
