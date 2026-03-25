# SignBridge

SignBridge is an end-to-end platform for translating text into sign language representations. The application consists of a modern web interface, an AI-powered text-to-sign inference backend, and an API services layer to query and stream pre-generated sign language video datasets.

## Project Structure

This repository is split into several interconnected services:

- `frontend/`: A Next.js web application providing the user interface. Built with React, TailwindCSS, Framer Motion, and Three.js for rendering sign animations.
- `backend/`: A FastAPI Python backend that hosts the machine learning inference pipeline (`SignModel`) for text-to-sign translation.
- `sign-idd-api/`: A FastAPI microservice designed to serve, search, and stream pre-generated sign language videos and text indices.
- `SignIDD_CodeFiles/`: Contains the core diffusion model architecture (Sign-IDD), training logic, and evaluation scripts. *(Note: Out-of-the-box inference for custom text is currently not supported without the underlying dataset dictionaries and a dedicated `predict.py` script).*
- `sign_idd_model_20260121_171210/`: Contains the large model checkpoints (e.g., `best.ckpt`), PyTorch skeleton tensors, configuration files (`Sign-IDD.yaml`), and generated video files.

---

## Prerequisites

Before setting up the project, make sure you have the following installed:
- **Node.js** (v18 or higher) & **npm**
- **Python** (3.9 or higher)
- **Git**

---

## 🚀 Setup & Installation Guide

### 1. Frontend Setup (Next.js)
The frontend relies on Supabase for data and authentication.

```bash
cd frontend

# Install Node dependencies
npm install

# Create a local environment file and add your Supabase keys
cp .env.local.example .env.local
# Or manually create .env.local and populate NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY

# Start the development server (runs on http://localhost:3000)
npm run dev
```

### 2. Backend Setup (Inference API)
This service exposes the `/translate` endpoint matching text requests with the ML model.

```bash
cd backend

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required Python dependencies (PyTorch, FastAPI, etc.)
pip install -r requirements.txt

# Start the main backend server (runs on http://localhost:8000)
uvicorn main:app --reload
```

### 3. Sign-IDD API Setup (Video serving API)
This secondary API serves the pre-generated sign-language mappings and videos.

```bash
cd sign-idd-api

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure model path (Optional)
# The API expects the model folder at `../sign_idd_model_20260121_171210` by default.
# You can change this by editing `config.py` in the sign-idd-api folder 
# or by setting the environment variable:
# export MODEL_DIR="/path/to/your/custom_model_folder"

# Start the API on a separate port (e.g., 8001) to avoid conflicts
uvicorn main:app --reload --port 8001
```

---

## Important Notes
- **Model Files & `.gitignore`:** The `/sign_idd_model_20...` directory contains heavy checkpoint files (`.ckpt`), tensor files (`.pt`), tracking information, and dynamically generated files like `lookup.json` or generated `videos/`. These need to be present locally but are strictly ignored in `.gitignore` due to size limitations. Their lookup path is easily configurable in `sign-idd-api/config.py`.
- **Dependencies:** The backend uses heavy libraries (like `torch` and `numpy`). A dedicated virtual environment is strongly recommended to isolate dependencies.
- **Supabase Integration:** Make sure you configure your `.env.local` accurately using your personal or project-wide Supabase credentials to access the DB and auth services natively.

## License
*Placeholder - refer to individual directory licenses or the overarching project license.*
