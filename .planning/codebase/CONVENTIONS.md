# Conventions

This document captures the expected coding standards, naming conventions, and patterns observed in the codebase.

## Frontend Styles
- **TypeScript:** Preferred over JS. Component definitions require typing for prop structures.
- **Styling:** TailwindCSS inline utility patterns are standard. `clsx` and `tailwind-merge` (`twMerge`) are used to logically combine Tailwind classes programmatically.
- **Animations:** Micro-animations happen via Framer Motion. Procedural web animations (more complex logic) prefer GSAP. Model visualization is routed natively to Three.js through React-Three-Fiber.

## API & Backend Styles
- **Framework Implementation:** FastAPI acts as the canonical backend.
- **Data Validation:** Route handlers must define input/output structures using Pydantic `BaseModel` classes defined in `schemas/`.
- **Environment Management:** Each Python microservice runs its own isolated virtual environment (`venv/`) and maintains an exact `requirements.txt`.
- **Error Handling:** Usage of FastAPI's `HTTPException(status_code=500, detail=...)` in try-catch loops is standard protocol for server failures, ensuring predictable error schemas to connected clients.
- **Startup Hooks:** Startup mechanisms like `@app.on_event("startup")` process heavy data actions (like index loaders) before the app binds ports, ensuring memory states are fully loaded globally.
