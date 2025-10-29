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

## 🧩 System Architecture — Field & Admin Clients (Extended)

This hybrid system extends beyond standard recognition by integrating a **two-tier verification model** designed for border-control-style deployments.

### 🧠 Field Client (On-Site Recognition)

- Detects and matches faces in real-time using **NeuralHash + HDIC** fusion.
- Captures multiple frames and performs **majority-based matching**.
- Automatically sends **best-match alerts** (image, ID, score, timestamp) to the **Admin Client** for manual confirmation.
- Provides **audio and visual feedback** for matches and non-matches.
- Displays recent verification feedback received from Admin.

### 🧍‍♂️ Admin Client (Central Command Center)

- Receives **manual verification requests** from Field Clients.
- Displays captured face, predicted ID, and match score for operator review.
- Admin can **accept** (confirm identity) or **reject** (deny match).
- Feedback is sent back to Field Clients in real-time and logged locally.
- Manages watchlist database (enroll, remove, configure thresholds).

---

## 📂 Structure

```bash
Hybrid_Face_Recognition/
├── cli/                  # CLI utilities
├── common/               # Shared modules
├── preprocess/           # MTCNN-based face alignment
├── neuralhash/           # NeuralHash pipeline
├── hdic/                 # HDIC hypervector encoding
├── fusion/               # Fusion algorithms (cascade / parallel)
├── db/                   # JSONL watchlists
├── software_builds/
│   ├── field_client/     # Field checkpoint system
│   │   ├── backend/      # FastAPI + matching backend
│   │   └── ui/           # React + Tailwind frontend
│   └── admin_client/     # Central admin interface
│       ├── backend/      # FastAPI backend for manual checks
│       └── ui/           # React + Tailwind admin dashboard
└── dataset/              # Images for enrollment/testing


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

### 4. Software Systems (Field & Admin Clients)

Beyond CLI tools, this hybrid system includes two interactive software applications for deployment and supervision:

### i. Field Client Software

The **Field Client** performs real-time face recognition using webcam feeds.  
It integrates the **Parallel NH + HDIC** fusion engine with visual and audio feedback.

🚀 Start Admin Backend:

From the existing virtual environment, run:

```bash
cd software_builds/field_client/backend
uvicorn main:app --reload --port 5001
```

💻 Start Field Client UI:

Open another terminal and run;

```bash
cd software_builds/field_client/ui
npm install
npm run start
```

### ii. Admin Client Software

The Admin Client acts as a central verification console.
It receives alerts from all field clients and lets authorized officers manually confirm or reject them.

🚀 Start Admin Backend:

From the same virtual environment:

```bash
cd software_builds/admin_client/backend
uvicorn admin_api:app --reload --port 5002
```

💻 Start Admin UI:

Open another terminal and run;

```bash
cd software_builds/admin_client/ui/admin_ui
npm install
npm run dev
```

---

## 🔁 Field–Admin Communication Flow

```bash
[ Field Camera ]
     ↓
  Detection & Fusion (NH + HDIC)
     ↓
  MATCH → Best Frame Sent → [ Admin Backend ]
                                  ↓
                    Manual Review → Accept / Reject
                                  ↓
           Feedback Returned → [ Field Client UI ]
```
