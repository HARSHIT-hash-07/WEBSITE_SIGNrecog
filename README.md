# SignBridge: Real-Time AI Sign Language Translation 🌍🤟

SignBridge is a high-performance, end-to-end platform for translating text into fluid, 3D sign language animations. It integrates a **state-of-the-art Diffusion Model (Sign-IDD)** to generate life-like skeletal motion from natural language.

---

## 🏛️ Project Architecture

The platform is now powered by a **Hybrid Cloud Architecture**:

- **`frontend/`**: Next.js (React) application. Features a 3D avatar bridge built with Three.js and Framer Motion. **Now connected to the Cloud Inference Engine.**
- **`hf_deploy/`**: The **Cloud Backend**. A containerized FastAPI server deployed to Hugging Face Spaces. It hosts the shrunken, optimized Sign-IDD Diffusion Engine (442MB).
- **`SignIDD_CodeFiles/`**: The core research implementation of the Sign-IDD architecture (ACD Diffusion, Encoder, Denoiser).
- **`sign-idd-api/`**: A legacy microservice for searching pre-generated PHOENIX14T dataset videos.

---

## 🚀 Getting Started (Quick Start)

### 1. Frontend Setup (The Bridge)
The frontend is pre-configured to talk to the Hugging Face Cloud Engine. You don't need a local backend to run translations!

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to see it in action.

---

## 🧠 Cloud Inference Engine (Sign-IDD)

Unlike simple video lookup systems, SignBridge uses **A-GPS (Sign-IDD)** to generate *new* motion.

### How it works:
1.  **Text-to-Gloss**: English text is mapped to GSL glosses (e.g., "Today rainy" -> "HEUTE REGEN").
2.  **Cloud Diffusion**: The request hits the [Hugging Face Space](https://huggingface.co/spaces/Harshit2907/sign-idd-inference).
3.  **Optimized Sampling**: The cloud engine runs 20-50 steps of DDIM sampling using **Half-Precision (float16)** weights for 2x faster results.
4.  **Instant Streaming**: The backend generates and returns a web-optimized MP4 video representing the sign's fluidity.

---

## 🛠️ Deployment & Model Optimization

To fit the **1.14 GB** model into Hugging Face's **1 GB** limit, we performed several optimizations:

1.  **Weight Stripping**: Removed 300MB+ of training-only "luggage" (optimizer states, gradients, schedulers).
2.  **Half-Precision (FP16)**: Converted 32-bit weights to 16-bit, reducing the size to **442 MB** without losing visual fidelity.
3.  **Containerization**: Built a custom `Dockerfile` with FFmpeg and OpenCV for cloud-native video rendering.

### Deploying the Cloud Engine:
If you modify the backend logic in `hf_deploy/`, redeploy with:
```bash
cd hf_deploy
git add .
git commit -m "Update backend engine"
git push origin main
```

---

## 🛠️ Local Development (Optional)
If you wish to run the backend locally:
1.  Navigate to `hf_deploy/`.
2.  Install requirements: `pip install -r requirements.txt`.
3.  Run: `uvicorn backend.main:app --port 8000`.
4.  Update `frontend/components/features/TextToSignClient.tsx` to point to `localhost:8000`.

---

## 📊 Dataset Attribution
This project is built using the **PHOENIX-2014-T** dataset and the **Sign-IDD (Ankita et al.)** diffusion architecture.

---

## 📄 License
Refer to the individual service directories for specific licensing details.
