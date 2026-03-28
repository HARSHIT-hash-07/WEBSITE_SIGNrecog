# Integrations

This document tracks all external APIs, internal network communications, state databases, and third-party SaaS tools used by the SignBridge system.

## 1. Third-Party Services
- **Supabase:** Used natively on the frontend via `@supabase/ssr` and `@supabase/supabase-js`. Provides:
  - **Authentication:** Managing user login and sessions.
  - **Database Storage:** Storing user preferences, lookup states, and application content.

## 2. Internal Microservice Communication
The main web UI does not process ML models directly; instead, it delegates to internal local services.
- **Inference API (Port 8000):** 
  - Called by the Frontend to request text-to-sign generation via `POST /translate`. 
  - Returns `TranslationResponse` containing skeleton mappings and video URLs.
- **Sign-IDD API (Port 8001):** 
  - Accessed by the Frontend to retrieve pre-generated video sequences and lookup endpoints.
  
## 3. External Networking Rules
- **CORS:** 
  - The Inference Backend explicitly maps origins to `http://localhost:3000` and `http://127.0.0.1:3000`.
  - The Sign-IDD API sets `allow_origins=["*"]`, allowing open querying from any interface.
