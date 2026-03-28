# Tech Stack

This document outlines the core languages, runtimes, frameworks, dependencies, and configuration for the SignBridge project.

## 1. Frontend Web Development
The frontend is built as a modern, high-performance web application utilizing the React ecosystem.
- **Framework:** Next.js (App Router, v16.1.6)
- **Language:** TypeScript 
- **Styling:** TailwindCSS with clsx and tailwind-merge
- **UI Components:** Radix UI primitives, shadcn UI conventions
- **State Management:** Zustand
- **Animations & 3D:** Framer Motion, GSAP, React Three Fiber (`@react-three/fiber`), Three.js

## 2. Core Backend (Inference API)
This microservice hosts the machine learning inference logic for Text-to-Sign translation.
- **Framework:** FastAPI
- **Language:** Python
- **Server:** Uvicorn
- **Machine Learning & Math:** PyTorch (`torch`), NumPy, Pandas, SciPy, Einops
- **Utilities:** tqdm, pyyaml, matplotlib

## 3. Serving Backend (Sign-IDD API)
This microservice serves and streams pre-generated sign language videos.
- **Framework:** FastAPI
- **Language:** Python
- **Server:** Uvicorn (`uvicorn[standard]`)
- **Data Validation:** Pydantic
- **File Parsing:** python-multipart
