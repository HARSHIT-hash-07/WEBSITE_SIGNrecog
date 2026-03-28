# SignBridge: Real-Time AI Sign Language Translation 

SignBridge is a high-performance, end-to-end platform for translating text into fluid, 3D sign language animations. It integrates a **state-of-the-art Diffusion Model (Sign-IDD)** to generate life-like skeletal motion from natural language.

---

## 🏛️ Project Architecture

The repository is organized into four core pillars:

- **`frontend/`**: Next.js (React) application. Features a 3D avatar bridge built with Three.js and Framer Motion for high-fidelity animation playback.
- **`backend/`**: The **Inference Powerhouse**. A FastAPI server hosting the Sign-IDD Diffusion Engine. Optimized for Apple Silicon (M2/M3) and NVIDIA GPUs.
- **`SignIDD_CodeFiles/`**: The core research implementation of the Sign-IDD architecture, including the ACD (Diffusion), Encoder, and Denoiser modules.
- **`sign-idd-api/`**: A lightweight microservice for searching and streaming the PHOENIX14T pre-generated dataset videos.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Node.js** v18+
- **Python** 3.9+ (3.11 recommended)
- **Git LFS** (Optional, for large file management)
- **Hardware**: Apple M1/M2/M3 (MPS supported) or NVIDIA GPU (CUDA supported).

### 2. The AI Brain (Model Setup)
The Sign-IDD model weights (`best.ckpt`) are >1GB and are not stored in Git.
1. Download the `best.ckpt` file from the project Google Drive/Source.
2. Place it in a folder named `sign_idd_model_20260121_171210/` at the project root.
3. **Verify the DNA**: The repo includes `backend/model_configs/src_vocab.txt`. This 1,089-word dictionary is hard-coded to your model's embedding layer (indices must match perfectly).

### 3. Backend Setup (The Engine)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the translation engine
uvicorn main:app --reload
```
*The engine will automatically detect if you are on a Mac (MPS) or a PC (CUDA) and optimize the diffusion math accordingly.*

### 4. Frontend Setup (The Bridge)
```bash
cd frontend
npm install
cp .env.local.example .env.local  # Add your Supabase keys here
npm run dev
```

---

## 🧠 Real-Time Sign Diffusion (Sign-IDD)

Unlike simple video lookup systems, SignBridge uses **A-GPS (Sign-IDD)** to generate new motion. When you send a request to `/translate`:

1. **Text-to-Gloss**: Your English text is mapped to German Sign Language (GSL) glosses (e.g., "Today rainy" -> "HEUTE REGEN").
2. **Diffusion Sampling**: The model runs 20-50 steps of DDIM sampling to "draw" a 3D skeleton in space, representing the sign's fluidity.
3. **Skeleton Stream**: The backend returns a JSON sequence of `(frames, 50, 3)` coordinates.
4. **3D Playback**: The Next.js frontend receives these points and uses Three.js to manipulate the avatar's joints in real-time.

---

## 🛠️ Reproducibility & Troubleshooting

- **MPS/Float64 Error**: If running on Mac M2, the system automatically casts to `float32` to ensure compatibility with Metal Performance Shaders.
- **Vocabulary Mismatch**: Do NOT modify `backend/model_configs/src_vocab.txt`. The indices (0-1088) are strictly tied to the `best.ckpt` embedding layer.
- **Bootstrapping**: If you need to rebuild the vocabulary from the official NSLT source, run `python backend/build_vocab.py`.

---

## 📊 Dataset Attribution
This project is built using the **PHOENIX-2014-T** dataset and the **Sign-IDD (Ankita et al.)** diffusion architecture.

---

## 📄 License
Refer to the individual service directories for specific licensing details.
