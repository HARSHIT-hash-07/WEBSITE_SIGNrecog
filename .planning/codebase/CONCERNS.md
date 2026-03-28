# Concerns

This document summarizes known architectural issues, technical debt, and areas of fragility within the repository.

## 1. Technical Debt & Edge Cases
- **Missing Inference Engine Capabilities:** Core Inference for truly out-of-the-box text generation is limited without underlying dataset dicts/lookup logic, representing a hurdle to freeform typing on the UI.
- **Testing Architecture:** The complete absence of formal testing frameworks (no Jest/Pytest) makes future regression bugs much more likely.
- **Missing CI/CD Pipeline:** No continuous integration configured to manage these distinct multi-lingual services.
- **Port Conflicts:** The project assumes strict adherence to ports 8000, 8001, and 3000 locally.

## 2. Fragile System Operations
- **Static Output Coupling:** Heavy coupling to serving static `/output` directly from runtime process instances.
- **Memory Consumption:** Loading enormous weights (`.pt` or `.ckpt` tensors) asynchronously into FastAPI apps at startup might trigger OOM (Out of Memory) conditions on constrained environments.
- **File Exclusions:** Core `.gitignore` configuration restricts committing massive generative assets from `/sign_idd_model_...`. These must be manually re-established on new environments, leading to potential initialization faults.

## 3. Security Focus
- Supabase environment injections inside frontend `NEXT_PUBLIC` variables must be properly controlled to prevent exposing elevated DB privileges.
- CORS on FastAPI endpoints defaults `allow_origins=["*"]` on the Sign-IDD API layer and hardcoded local hosts (`http://localhost:3000`) on the Backend Layer. This poses environment staging limitations.
