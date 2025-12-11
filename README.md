# Emotion Classification API + Browser Extension

Multi-label emotion classification using **DeBERTa-v3-large + HEF** (Hand-crafted Emotion Features).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINER                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (app.py)                            │   │
│  │  + Model (best_model.pt) - ~1.7GB                   │   │
│  │  + Thresholds (ensemble_thresholds.npy)             │   │
│  │  + Python dependencies (torch, transformers, etc.)  │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↑                                   │
│                    Port 8000                                │
└─────────────────────────────────────────────────────────────┘
                          ↑
                   HTTP POST /predict
                          ↑
┌─────────────────────────────────────────────────────────────┐
│         BROWSER EXTENSION (NOT in Docker)                   │
│  - Installed manually in Chrome/Firefox                     │
│  - Scans Facebook/Messenger/Zalo messages                   │
│  - Calls API: http://localhost:8000/predict                 │
│  - Displays emotion badges on messages                      │
└─────────────────────────────────────────────────────────────┘
```

**Important:** The browser extension is NOT containerized - it runs inside your browser.

---

## Project Structure

```
emotion_api/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker orchestration
├── requirements.txt        # Python dependencies
├── app.py                  # FastAPI server
├── model_utils.py          # Model inference utilities
├── models/
│   ├── best_model.pt       # Trained model (~1.7GB)
│   └── ensemble_thresholds.npy
└── extension/              # Browser extension (COPY THIS TO USE)
    ├── manifest.json       # Extension config
    ├── content.js          # Main logic
    └── style.css           # Styling
```

---

## How to Use This Project

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Git LFS** installed (for downloading model files)

```bash
# Install Git LFS (if not already installed)
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Ubuntu: sudo apt install git-lfs

git lfs install
```

### Step 1: Clone and Run Docker

```bash
# Install Git LFS first (required for model files)
git lfs install

# Clone the repository (~1.7GB model files will download automatically)
git clone https://github.com/anhtp35/NLP_sentiment.git
cd NLP_sentiment

# Build and run with Docker Compose
docker-compose up --build
```

**First run takes ~5-10 minutes** (downloading DeBERTa backbone + loading model)

### Step 2: Verify API is Running

Open browser: **http://localhost:8000/docs**

Or test with curl/PowerShell:
```bash
# Linux/Mac
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'

# Windows PowerShell
$body = @{text="I am so happy today!"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

Expected response:
```json
{
  "original": "Hôm nay tôi rất vui!",
  "translated": "I am so happy today!",
  "emotions": [
    {"label": "joy", "score": 0.92},
  ]
}
```

### Step 3: Install Browser Extension

1. **Copy the `extension/` folder** from Docker container or project
2. Open Chrome → `chrome://extensions/`
3. Enable **Developer mode** (top right toggle)
4. Click **Load unpacked**
5. Select the `extension/` folder
6. Go to **facebook.com** or **messenger.com** or **chat.zalo.me**
7. You'll see emotion labels on messages! 🎉

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict emotions (extension format) |
| `/predict/full` | POST | Predict with full details |
| `/health` | GET | Health check |
| `/emotions` | GET | List 28 supported emotions |
| `/docs` | GET | Swagger UI documentation |

---

## Docker Commands

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---


