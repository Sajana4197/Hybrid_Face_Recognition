# Hybrid Face Recognition (NeuralHash + HDIC)

🚀 A hybrid face recognition system combining **Apple’s NeuralHash** (fast shortlist) and **Hyperdimensional Image Classification (HDIC)** (robust confirmation).  
Ideal for high-security applications like border control.

---

## 🔑 Features

- NeuralHash → per-image 96-bit hashes
- HDIC → 10,000D hypervector prototypes (3 clusters per person)
- Cascade matching (fast shortlist + strong confirmation)
- JSONL database storage
- CLI tools: `enroll`, `match`

---

## 📂 Structure

```bash
hybrid_face_rec/
├── cli/          # CLI tools
├── common/       # Utilities
├── preprocess/   # Face alignment (MTCNN)
├── neuralhash/   # NeuralHash pipeline
├── hdic/         # HDIC pipeline
├── db/           # Databases
├── dataset/      # Images
```

## ⚙️ Install

To install required packages run

```bash
pip install -r requirements.txt
```

## 📝 Usage

To enroll a new person, run

```bash
python -m cli.enroll --id n000002 --name "John Doe" --images "dataset/n000002"
```

and for a comparison, run

```bash
python -m cli.match --image dataset/probe/unknown1.jpg --K 5
```
