# Structure

## Directory Layout & Key Locations

The top-level structure designates separate environments per service, reflecting the Microservice Architecture.

### 1. `/frontend/`
Typical Next.js scaffolding.
- `app/`: Next.js App Router definitions, page tree.
- `components/`: UI elements, segmented appropriately (typically layout, ui, forms).
- `lib/`: Standard utilities, types, wrapper functions.
- `utils/`: Includes logic like Supabase client initiations and middleware.
- `public/`: Static file serving for the frontend.

### 2. `/backend/`
- `main.py`: The main entry point for the FastAPI server. Configures routes, mounts `/static`.
- `model_loader.py`: Contains the `SignModel` class and handles model weights, dependencies, and tensor transformation.
- `verify_api.py`: Optional API interaction validator.
- `output/`: Ephemeral or persistent storage dynamically created for output generation.

### 3. `/sign-idd-api/`
- `main.py`: Entry point for serving static structures. Uses standard routing inclusions.
- `routers/`: Controller equivalents for handling path specifics (e.g. `videos.py`).
- `schemas/`: Pydantic object models bridging I/O validation (e.g., `TranslationRequest`).
- `utils/`: Data loaders logic, notably the file index mapper.
- `config.py`: Exposes runtime paths via OS environments.

### 4. Machine Learning Cores
- `/SignIDD_CodeFiles/`: Raw code underlying the diffusion modelling (the "science" environment).
- `/sign_idd_model_20260121_171210/`: State checkpoints, large dictionaries, lookup logic representing the compiled weight state of the system locally.
