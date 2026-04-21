

# SignBridge: AI-Powered Sign Language Translation Platform

**SignBridge** is a high-performance, end-to-end platform designed to bridge the communication gap for the Deaf and hard-of-hearing community.

---

## ⚡ Quick Run (How to start)

To launch the platform locally, run these in two separate terminal windows:

### 1. Start AI Backend
```bash
cd hf_deploy && source venv/bin/activate && uvicorn backend.main:app --port 8000 --reload
```

### 2. Start Frontend Website
```bash
cd frontend && npm run dev
```

---

## 🏛️ System Architecture

SignBridge is engineered with a **Hybrid Cloud Architecture** to ensure high performance on any device without requiring local GPU resources.

- **Frontend:** Built with **Next.js 14**, **TypeScript**, and **Framer Motion**. Deployed on Vercel.
- **AI Backend:** A **FastAPI** inference server hosting the **Sign-IDD Diffusion Engine**. Deployed on Hugging Face Spaces.
- **Database:** **Supabase (PostgreSQL)** for user authentication, search history, favorites, and feedback.
- **Inference Engine:** Uses DDIM sampling and **PINN (Physics-Informed Neural Network)** losses to ensure anatomically correct and smooth motion.

---

## ✨ Key Features

- **Real-Time Generation:** Converts text to sign language motion in seconds.
- **Cloud-Native Inference:** Optimized model (1.14GB $\to$ 442MB) fits into free-tier cloud environments.
- **User Ecosystem:** Secure Auth, Search History, Favorites, and Feedback systems built-in.
- **Premium UI:** Dark-mode first design with glassmorphism and tech-forward aesthetics.
- **Cross-Platform:** Works seamlessly on Desktop and Mobile browsers.

---

## 🚀 Local Setup Guide

Follow these steps to run the complete SignBridge environment on your local machine (Windows, macOS, or Linux).

### 📋 Prerequisites

Ensure you have the following installed:
- **Node.js** (v18.x or higher)
- **Python** (3.9 - 3.11 recommended)
- **Git**
- **FFmpeg** (Required for video rendering)

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/YourUsername/SignBridge.git
cd SignBridge
```

### Step 2: Configure Environment Variables
Create a file named `.env.local` in the project root and add your keys:
```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key

# Backend Configuration
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

### Step 3: Setup the AI Backend (FastAPI)

#### **On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r hf_deploy/requirements.txt
```

#### **On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r hf_deploy/requirements.txt
```

#### **Download the Model:**
Ensure your optimized model `.pth` file is placed in `hf_deploy/models/sign_idd.pth`.

#### **Run the Backend:**
```bash
cd hf_deploy
uvicorn backend.main:app --port 8000 --reload
```
*The backend is now live at `http://localhost:8000`.*

---

### Step 4: Setup the Frontend (Next.js)

Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
*The website is now live at `http://localhost:3000`.*

---

## 🛠️ Model Optimization (The "Shrink" Strategy)

To deploy a heavy **1.14 GB PyTorch model** on free-tier cloud hosting (Hugging Face / Vercel), we implemented a custom optimization pipeline:

1.  **Optimizer Stripping:** Removed training-only metadata and Adam optimizer states (Saved ~300MB).
2.  **Quantization (FP16):** Converted 32-bit weights to 16-bit half-precision (Saved ~400MB).
3.  **Async Loading:** Implemented a Singleton threading pattern to prevent "cold start" API timeouts.

Final Deployment Size: **442 MB** (61% reduction).

---

## 📊 Dataset & Research Attribution

- **Dataset:** [PHOENIX-2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) — German Sign Language Weather Broadcasts.
- **Architecture:** Based on the **Sign-IDD** research paper (Ankita et al.).
- **Vocabulary:** 1,089 unique sign language tokens reconstructed for this implementation.

---

## 📄 License

This project is for academic and research purposes. Please refer to the specific licenses in the `frontend/` and `hf_deploy/` subdirectories for third-party library details.

---

**Developed with ❤️ by Harshit (2026)**
*"Empowering the world through accessible AI."*
