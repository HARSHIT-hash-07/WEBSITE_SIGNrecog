
---

# EXPLORATORY PROJECT REPORT

## SignBridge: An AI-Powered Sign Language Translation Platform

---

| | |
|---|---|
| **Student Name** | Harshit Vaghamshi |
| **Roll Number** | 24075091 |
| **Programme** | B.Tech |
| **Institution** | Indian Institute of Technology (BHU), Varanasi |
| **Date of Submission** | 19 April 2026 |

---

## DECLARATION

I hereby declare that the work presented in this Exploratory Project Report titled *"SignBridge — An AI-Powered Sign Language Translation Platform"* is an original piece of work carried out by me under my own initiative as part of my academic explorations at IIT (BHU), Varanasi. This report has not been submitted for any other degree, diploma, or qualification at any institution. All sources of information, datasets, libraries, and research papers that have been referenced or used in this project are duly acknowledged in the References section.

**Harshit Vaghamshi**
Roll No.: 24075091 | B.Tech
IIT (BHU), Varanasi
19 April 2026

---

## ABSTRACT

Approximately 430 million people worldwide communicate primarily through sign languages — rich, grammatically complete visual-gestural languages that remain poorly supported by modern computational systems. While automatic speech recognition and text-to-speech technologies have matured significantly, the problem of automatically *generating* sign language from text — Sign Language Production (SLP) — remains an open challenge, confined largely to academic research.

This report presents **SignBridge**, an end-to-end, AI-powered web platform that performs text-to-sign language translation, converting written English sentences into animated three-dimensional skeletal sequences representing German Sign Language (Deutsche Gebärdensprache, DGS). The system is built around the **Sign-IDD** (Sign Language via Implicit Diffusion Denoising) model — a state-of-the-art generative architecture that combines a multi-rate Denoising Diffusion Implicit Model (DDIM) with Physics-Informed Neural Network (PINN) regularisation to produce anatomically plausible, temporally coherent sign sequences. The model is trained on the PHOENIX-2014-T dataset of German weather-forecast sign language broadcasts, spanning a vocabulary of 1,089 sign glosses.

Beyond the AI core, SignBridge is a complete production system: a Python FastAPI inference server containerised with Docker and deployed on Hugging Face Spaces, a Next.js 14 TypeScript frontend deployed on Vercel, and a Supabase PostgreSQL backend supporting user authentication, search history, personalised favourites, and translation feedback. A key engineering contribution of the project is a model compression pipeline that reduced the trained checkpoint from 1.14 GB to 442 MB — a 61% reduction — enabling deployment on free-tier cloud infrastructure without compromising inference quality. The resulting system is publicly accessible in a browser on any device, requiring no local GPU or installation.

---

## TABLE OF CONTENTS

- [Chapter 1 — Introduction](#chapter-1--introduction)
  - [1.1 Problem Statement and Motivation](#11-problem-statement-and-motivation)
  - [1.2 Objectives of the Project](#12-objectives-of-the-project)
  - [1.3 Scope and Delimitations](#13-scope-and-delimitations)
  - [1.4 Organisation of the Report](#14-organisation-of-the-report)
- [Chapter 2 — Literature Review and Background](#chapter-2--literature-review-and-background)
  - [2.1 Sign Language: Linguistic and Computational Perspective](#21-sign-language-linguistic-and-computational-perspective)
  - [2.2 Existing Approaches to Sign Language Production](#22-existing-approaches-to-sign-language-production)
  - [2.3 Generative Modelling with Diffusion Models](#23-generative-modelling-with-diffusion-models)
  - [2.4 The Sign-IDD Research Architecture](#24-the-sign-idd-research-architecture)
  - [2.5 The PHOENIX-2014-T Dataset](#25-the-phoenix-2014-t-dataset)
- [Chapter 3 — System Design and Architecture](#chapter-3--system-design-and-architecture)
  - [3.1 Architectural Philosophy and Design Rationale](#31-architectural-philosophy-and-design-rationale)
  - [3.2 Three-Tier System Architecture](#32-three-tier-system-architecture)
  - [3.3 End-to-End Data Flow](#33-end-to-end-data-flow)
  - [3.4 Technology Selection Rationale](#34-technology-selection-rationale)
  - [3.5 Deployment Architecture](#35-deployment-architecture)
- [Chapter 4 — Implementation](#chapter-4--implementation)
  - [4.1 AI Inference Engine](#41-ai-inference-engine)
  - [4.2 Backend API Service](#42-backend-api-service)
  - [4.3 Model Compression and Cloud Deployment](#43-model-compression-and-cloud-deployment)
  - [4.4 Frontend Web Application](#44-frontend-web-application)
  - [4.5 Database Design and User Ecosystem](#45-database-design-and-user-ecosystem)
- [Chapter 5 — Results and Discussion](#chapter-5--results-and-discussion)
  - [5.1 Functional Outcomes](#51-functional-outcomes)
  - [5.2 Model Compression Results](#52-model-compression-results)
  - [5.3 Deployment Status](#53-deployment-status)
  - [5.4 Discussion of Output Quality](#54-discussion-of-output-quality)
- [Chapter 6 — Conclusion and Future Work](#chapter-6--conclusion-and-future-work)
  - [6.1 Summary and Learnings](#61-summary-and-learnings)
  - [6.2 Future Directions](#62-future-directions)
- [References](#references)

---

## CHAPTER 1 — INTRODUCTION

### 1.1 Problem Statement and Motivation

Sign language is not a simplified or coded form of spoken language. It is an independent natural language with its own grammar, syntax, spatial morphology, and pragmatic conventions. German Sign Language (Deutsche Gebärdensprache, DGS), for example, uses topic-comment sentence structures, spatial referencing (where locations in the signing space represent entities), and non-manual markers — facial expressions and head position — that carry grammatical meaning such as negation, question type, and emphasis. These properties make sign language fundamentally different from spoken German, and its computational modelling a distinct and non-trivial research problem.

The global population of Deaf and hard-of-hearing individuals is estimated by the World Health Organisation at over 430 million, the majority of whom rely on sign languages as their primary mode of communication. Despite this, sign language is one of the most underserved modalities in computational linguistics. Technologies that have transformed access for hearing users — voice assistants, automatic transcription, machine translation, text-to-speech — have no reliable equivalent for sign language users. The result is a persistent communication equity gap: a Deaf person navigating a public information system, an educational resource, or a medical consultation is fundamentally more disadvantaged than their hearing counterpart.

This project is motivated by the conviction that AI-driven sign language translation, even in a research and proof-of-concept form, can contribute meaningfully to closing this gap. The core question the project addresses is: *can a state-of-the-art generative AI model for sign language production be successfully deployed as an accessible, browser-based tool for any user, anywhere?*

### 1.2 Objectives of the Project

The project was undertaken with the following specific objectives:

1. **To understand, implement, and adapt a state-of-the-art Sign Language Production (SLP) model** — specifically the Sign-IDD diffusion architecture — by studying the underlying research and engineering the model for inference rather than training.

2. **To build a complete, publicly accessible web platform** around the AI model, enabling any user with a browser to input text and receive a sign language animation without installing any software or requiring GPU hardware.

3. **To design and implement a production-grade inference pipeline** encompassing HTTP API design, model loading strategy, video generation, and cloud containerisation.

4. **To engineer a model compression pipeline** that reduces the trained checkpoint to a size compatible with free-tier cloud hosting, making the project deployable without paid infrastructure.

5. **To create a complete user ecosystem** around the translation feature — including authentication, personalisation (favourites, history), and a feedback mechanism — demonstrating the platform as more than a research demo.

6. **To explore and document the engineering considerations** involved in bridging research-grade machine learning with production web software.

### 1.3 Scope and Delimitations

The project is scoped to German Sign Language (DGS) generation using the PHOENIX-2014-T dataset, which is drawn from weather forecast television broadcasts. This means the supported vocabulary is domain-specific — covering temporal expressions (days, months), meteorological terms, and compass directions — and is not a general-purpose sign language system. The project is intended as an academic exploration and demonstration, not as a clinically or linguistically certified accessibility tool.

A Sign-to-Text feature (sign language recognition from camera input) is included as a preliminary Beta interface, but the underlying recognition model is not implemented in this version; the interface demonstrates the intended user journey when recognition capability is added in future work.

### 1.4 Organisation of the Report

Chapter 2 reviews the relevant scientific and technical background, including the linguistic properties of sign language, existing computational approaches, diffusion generative models, and the specific Sign-IDD architecture. Chapter 3 presents the overall system design and architecture of SignBridge. Chapter 4 describes the detailed implementation of each component. Chapter 5 presents and discusses the results. Chapter 6 concludes the report and outlines directions for future work.

---

## CHAPTER 2 — LITERATURE REVIEW AND BACKGROUND

### 2.1 Sign Language: Linguistic and Computational Perspective

Sign languages are full natural languages, not invented codes or visual representations of spoken language. They are acquired as first languages by Deaf children born into Deaf families, and as second languages by hearing children born into such families. They exhibit all the linguistic properties of natural language — recursive syntax, morphological complexity, arbitrariness of the sign-meaning relationship, historical evolution — while exploiting the articulatory richness of the visual-gestural modality.

From a computational modelling perspective, sign language poses several challenges absent in text or speech:

**Spatial Grammar**: Unlike the linear temporal string of spoken language, sign language exploits the three-dimensional signing space around the signer's body. Entities introduced into discourse are assigned locations in this space, and subsequent reference to these entities is made by pointing to or directing signs toward those locations. Modelling this spatial grammar requires tracking the assignment and use of spatial loci across utterances.

**Simultaneity**: Sign language can express multiple semantic dimensions simultaneously. A single sign can simultaneously specify handshape (the lexical root), orientation (grammatical modification), and location (spatial grammar), while the face simultaneously expresses emotional tone, question type, or negation. This simultaneity has no direct parallel in the sequential structure of speech or text.

**Multi-Channel Coordination**: A complete signed utterance involves the coordinated motion of both hands, the wrists, forearms, upper arms, the body torso, the head, and the face. A generative model must produce all of these modalities in a coherent, temporally synchronised sequence.

**High Dimensionality**: Representing a human signer as a 3D skeleton with 50 joints (each with three XYZ coordinates) produces a 150-dimensional motion vector per frame. A sign language sentence at 25 frames per second (fps) may span 60 to 300 frames, yielding sequences of 9,000 to 45,000 numbers that must be generated jointly and coherently.

### 2.2 Existing Approaches to Sign Language Production

The first generation of computational sign language production systems used **video sprite concatenation**: a dictionary of pre-recorded video clips, each showing a single sign, was assembled into sequences by cutting and joining clips in the correct order. This approach is linguistically limited (vocabulary is bounded by the dictionary) and produces unnatural output — the transitions between concatenated clips are visually discontinuous because natural signing involves coarticulation, the smooth blending of one sign into the next based on the upcoming sign.

The second generation used **avatar-based animation** with CG characters posed using sign language notation systems. While this overcomes the video quality limitations, pose specification is typically manual, and generative synthesis (producing natural-looking motion automatically) remains limited.

The third generation applies **deep learning** to learn the mapping from text or gloss sequences to skeletal pose sequences end-to-end. Initial deep learning approaches applied sequence-to-sequence models (encoder-decoder Transformers) to this problem, producing continuous pose sequences. However, regressing directly to a mean output over many possible signing trajectories tends to produce "mean pose" outputs — statistically average motions that look unnaturally still or stiff, because the model averages over the space of plausible realisations rather than committing to any specific one. This is the fundamental limitation of deterministic regression in a multi-modal output space.

The insight that motivated the Sign-IDD architecture — and diffusion models for motion generation more broadly — is that generative modelling rather than deterministic regression is the right framing. A generative model learns the full distribution of plausible signing trajectories, and samples from this distribution at inference time, producing diverse, non-average outputs.

### 2.3 Generative Modelling with Diffusion Models

**Denoising Diffusion Probabilistic Models (DDPMs)**, introduced by Ho et al. (2020), define a generative process through the reversal of a Markovian forward process. The forward process gradually corrupts a clean data sample (a pose sequence) by adding small amounts of Gaussian noise over T time steps, until the sample is indistinguishable from pure Gaussian noise. The model is trained to reverse this corruption, predicting at each step the slightly less noisy version of the current sample. At inference time, generation begins from pure Gaussian noise and iteratively denoises over all T steps.

The key insight is that this reverse process can be learned by a neural network trained to predict, at each step, the noise that was added or equivalently the clean sample. Because the starting point is random Gaussian noise, every sample drawn from the model is different, producing the diversity absent in deterministic regression.

The limitation of DDPMs for practical applications is that inference requires running the denoising network T times (often T = 1000), making generation slow. **Denoising Diffusion Implicit Models (DDIMs)**, proposed by Song et al. (2020), address this by defining a non-Markovian sampling trajectory that achieves high-quality generation with far fewer steps — typically 20 to 100 — without retraining the model. This capability is central to the feasibility of SignBridge as a web application, where inference latency must be kept reasonable.

The application of diffusion models to motion generation is an active research area. Unlike image generation where the output is a fixed-size 2D grid of pixels, motion sequences are variable-length temporal sequences with structural constraints (physical joint hierarchies, continuity). The Sign-IDD model adapts the DDIM framework to this domain with several domain-specific innovations.

### 2.4 The Sign-IDD Research Architecture

The Sign-IDD architecture (from the paper "Sign Language Production via Sign-IDD", Ankita et al.) is designed specifically for gloss-conditioned sign language generation. Its design is motivated by the following observations about sign language motion that generic diffusion models for motion do not address:

**Observation 1: Hand and body joints carry different information densities.** In sign language, the hands are the primary articulators: handshape, hand orientation, and hand movement directly encode lexical content. The body torso, shoulders, and head serve largely as a reference frame and provide prosodic and grammatical information, but change more slowly and with less linguistic specificity than the hands. A model that treats all joints identically is suboptimal.

**Observation 2: Sign language motion must be physically plausible.** Generated joint sequences must respect the kinematic structure of the human body — limbs cannot change length across frames, joints cannot move faster than physically realistic speeds, and the global body configuration must be consistent across the sequence.

**Observation 3: Generation must be conditioned on a sequential linguistic input.** The target sign sequence must correspond to the input gloss sequence. This requires integrating a text encoder that processes the gloss sequence into a representation that guides the entire denoising process.

These observations lead to three key design decisions in Sign-IDD:

#### 2.4.1 Multi-Rate Diffusion Scheduling

Sign-IDD employs **two separate cosine beta noise schedules** — one for body joints and a separate, scaled schedule for hand joints. Specifically, the hand joint noise schedule uses a scaling factor less than 1.0, meaning the hands are corrupted at a *slower* rate during the forward process. Equivalently, hand joints start with less noise at each timestep than body joints.

During the reverse (generative) process, this means the hands are denoised with greater precision at each step, since their noise characteristics are modelled separately. The two schedules are combined by using binary channel masks (body mask and hand mask) that select which joint channels receive which schedule's parameters at each diffusion timestep. This fine-grained per-channel noise modelling is referred to as the **multi-rate diffusion** approach.

#### 2.4.2 Adaptive Conditional Diffusion (ACD) Denoiser

At each step of the reverse diffusion process, a denoiser network takes the current noisy pose estimate and predicts the clean pose. In Sign-IDD, this denoiser is a **Transformer-based attention network** — the ACD Denoiser — that can attend to both the current noisy pose and the encoded representation of the input gloss sequence. This conditioning mechanism ensures that the generated motion coherently follows the input text throughout the denoising process, not just as an initial condition.

The denoiser also receives time step embeddings indicating which step of the diffusion process it is operating at, allowing it to calibrate its predictions to the expected noise level at that step.

#### 2.4.3 Physics-Informed Neural Network (PINN) Losses

To enforce physical plausibility during training, Sign-IDD introduces four regularisation terms into the training loss, collectively called the PINN losses:

- **Bone Length Consistency Loss**: Penalises changes in the Euclidean distance between parent and child joints (bone lengths) across frames. Since bones do not stretch or compress in a living person, this loss enforces anatomical consistency across the full generated sequence.

- **Velocity Smoothness Loss**: Computes the per-joint velocity (first temporal derivative of position) and penalises large velocities. This discourages discontinuous, jerky motion between frames.

- **Acceleration Smoothness Loss**: Computes the per-joint acceleration (second temporal derivative) and penalises large accelerations. This enforces dynamically smooth motion, since humans accelerate and decelerate continuously, not instantaneously.

- **Forward Kinematics (FK) Consistency Loss**: For each bone, reconstructs the expected position of the child joint as the parent joint position plus the unit direction vector of the bone scaled by the rest-length. The difference between this reconstruction and the actual predicted child position is penalised, enforcing global kinematic chain coherence.

Each loss term is weight by a coefficient (λ_bone = 1.0, λ_vel = 0.1, λ_acc = 0.05, λ_fk = 0.5) and the total PINN term is weighted at 0.1 relative to the primary reconstruction loss. The Huber loss variant (δ = 1.0) is used in each component for robustness to outlier joint positions. All loss computations are mask-aware, excluding padded frames in variable-length sequences.

#### 2.4.4 Iconicity Expansion

Before a noisy pose estimate is passed to the ACD Denoiser, it is transformed from its raw 150-dimensional representation (50 joints × 3 XYZ coordinates) to a 350-dimensional representation (50 joints × 7 features) through an "iconicity" (ID) expansion. The seven features per joint are: the three raw XYZ coordinates, plus the unit direction vector (3 values — the normalised direction from each joint toward a reference point) and the bone length scalar. This richer feature representation provides the denoiser with explicit geometric context — not just where each joint is, but how it is oriented and how long the bone is — making the denoising task easier and the resulting predictions more geometrically consistent.

### 2.5 The PHOENIX-2014-T Dataset

The RWTH-PHOENIX-Weather 2014T (PHOENIX-2014-T) dataset is the benchmark standard for German Sign Language research. It was collected at the RWTH Aachen University from German public television sign language weather forecast broadcasts, providing a controlled, domain-specific, naturally produced sign language corpus.

The dataset provides:

- Approximately **7,096 training sentences**, 519 development sentences, and 642 test sentences.
- **Gloss-level annotations** — each sentence is annotated with the sequence of DGS sign glosses produced by the signer.
- **Sentence-level German text** corresponding to each signing utterance, enabling studies of text-to-gloss and gloss-to-text translation.
- **3D skeletal pose data** extracted from the original video recordings using pose estimation, enabling the supervised training of a pose generation model.
- A vocabulary of **1,089 unique sign glosses** in the DGS lexicon.

The dataset's focus on weather forecasts produces a bounded, repetitive vocabulary dominated by signs for days of the week, cardinal directions, weather conditions (rain, sun, snow, wind), and temperature descriptors. While this bounded scope makes PHOENIX-2014-T well-suited as a proof-of-concept training corpus, it limits direct generalisability to open-domain communication.

---

## CHAPTER 3 — SYSTEM DESIGN AND ARCHITECTURE

### 3.1 Architectural Philosophy and Design Rationale

The overarching design philosophy of SignBridge is **accessibility by default**: the system must work for any user, on any device, without any installation, setup, or local hardware requirement. This principle directly drives every major architectural decision.

An AI model that can only be used by those with access to a GPU-equipped computer, or who are comfortable with Python environments and command-line inference scripts, is not truly accessible — it reproduces the expert-user barrier in a new form. For a sign language tool specifically, where the primary users are members of the Deaf community who interact with technology on mobile browsers and consumer devices, this is a critical consideration.

The accessibility principle leads to a **client-server separation**: all AI computation happens on a server; the user's device need only run a JavaScript web application. This further requires that the server be hosted in the cloud, be accessible over standard HTTPS, and be capable of handling requests without the user configuring anything. The choice to use free-tier cloud infrastructure is deliberate — it demonstrates that this kind of accessible AI deployment is achievable without commercial infrastructure costs.

### 3.2 Three-Tier System Architecture

SignBridge is organised as a three-tier application:

**Tier 1 — Presentation (Frontend)**: A Next.js 14 TypeScript web application, deployed on Vercel. This tier is responsible for all user interaction: receiving text input, communicating with the backend, displaying animations, managing authentication state, and providing the search and discovery interface. The frontend has no direct access to AI model weights or inference logic; it communicates with the backend exclusively through HTTP API calls.

**Tier 2 — Application and AI (Backend)**: A Python FastAPI inference server, containerised with Docker and hosted on Hugging Face Spaces. This tier hosts the Sign-IDD model and implements the full inference pipeline — text processing, neural network forward pass, pose generation, and video rendering. It exposes a RESTful HTTP API to the frontend. A secondary, lighter FastAPI service (the `sign-idd-api`) runs alongside the primary backend and handles pre-generated video indexing and search, serving as a fast dictionary lookup for content that does not require live AI generation.

**Tier 3 — Data (Database)**: A Supabase-managed PostgreSQL database, hosted on Supabase's cloud infrastructure. This tier is responsible for user identity (authentication), and three application tables: search history, saved favourites, and translation feedback. The frontend communicates with Supabase directly using the Supabase JavaScript SDK; the backend does not need database access.

### 3.3 End-to-End Data Flow

To make the architecture concrete, consider the lifecycle of a single text-to-sign translation request through the system:

The user types a sentence into the text input on the frontend and selects the "Live AI Bridge" mode. The frontend JavaScript makes an HTTP POST request to the backend `/translate` endpoint, carrying the input text as a JSON body. The backend receives the request and routes it to the inference engine.

The inference engine first applies a **text-to-gloss mapping** step, converting the English input into the DGS gloss tokens that the model was trained on. The resulting gloss sequence is tokenised — special BOS and EOS tokens are prepended and appended — and passed to the **Transformer encoder**, which produces a dense vector representation of the gloss sequence. This encoder output serves as the conditioning signal for the denoiser.

The model then enters the **DDIM sampling loop**: starting from a random Gaussian noise tensor of shape (1, T, 150) — representing one batch of T frames and 150 joint coordinates — the ACD module iteratively denoises the tensor over 20 steps (Standard mode) or 60 steps (HQ mode), at each step querying the ACD Denoiser network which attends to the encoder representation and predicts the clean pose estimate. After all steps, the final denoised tensor is the generated sign language pose sequence.

The pose sequence is passed to the **video renderer**, which draws each frame as a 3D skeletal figure using Matplotlib and assembles the frames into an MP4 video using FFmpeg at 25 frames per second. The generated video file is saved to the server's static file directory, its URL is included in the HTTP response, and the responding JSON is returned to the frontend.

The frontend receives the response, extracts the `video_url`, and renders the video in the animation player panel on the right side of the interface. Simultaneously, the search query is asynchronously logged to the Supabase `search_history` table.

In the **Search Videos** mode — the simpler discovery pathway — the frontend queries the secondary `sign-idd-api` service with the text query. This service maintains an in-memory index of pre-generated video filenames (generated offline from the PHOENIX dataset) and returns fuzzy-matched results. The user selects a result and the corresponding video is streamed directly from the API to the frontend video player.

### 3.4 Technology Selection Rationale

Each technology in the stack was selected for specific reasons:

**FastAPI (Python)** was chosen for the backend because the Sign-IDD model is implemented in PyTorch, which is a Python library. FastAPI provides asynchronous request handling, automatic OpenAPI documentation, and Pydantic-based schema validation, making it straightforward to build a well-typed, documented API around the Python inference code. Its performance characteristics (Uvicorn ASGI server, async I/O) are appropriate for a server that performs long-running CPU-bound inference and serves static files.

**Next.js 14 (TypeScript, App Router)** was selected for the frontend because it combines the developer experience of React with production-grade features including server-side rendering, automatic code splitting, file-based routing, and seamless Vercel deployment. TypeScript ensures type safety across the component tree and API response handling.

**Supabase** was chosen as the database and authentication provider because it offers a managed PostgreSQL database, a built-in authentication service supporting OAuth providers (Google Sign-In), Row-Level Security for data access control, and a JavaScript SDK that allows direct client-to-database communication, eliminating the need to route authentication and data operations through the AI backend.

**Docker on Hugging Face Spaces** was chosen for AI backend deployment because Hugging Face Spaces provides a free-tier, publicly accessible compute environment specifically designed for AI model serving. Docker containerisation ensures that the exact Python environment, system dependencies (including FFmpeg), and application code are reproducibly deployed regardless of the host environment.

**Vercel** was chosen for frontend deployment because of its native Next.js integration, zero-configuration continuous deployment from GitHub, edge network distribution, and generous free-tier limits for web application hosting.

### 3.5 Deployment Architecture

The deployment architecture is **hybrid cloud**: different tiers of the application run on different cloud providers, chosen for their respective specialisations.

The frontend is deployed on **Vercel**, served from a global edge CDN for low-latency page loads worldwide. Any push to the main branch of the GitHub repository triggers an automatic Vercel deployment, providing continuous integration out of the box.

The AI backend is deployed on **Hugging Face Spaces** as a Docker Space. The Dockerfile specifies a Python 3.10 base image, installs all dependencies from `requirements.txt`, copies the application source, and starts the Uvicorn server on port 7860. The model weights (442 MB) are not bundled into the Docker image or committed to the repository; instead, the backend implements an **automated weight sourcing mechanism** that checks for the weight file in the persistent `/data` storage directory on startup, and downloads it from a private Hugging Face Model Hub repository (`ExploWebsite/SignBridge-Weights`) if absent, authenticated via a secret token (`HF_TOKEN`). This pattern separates model versioning from application versioning and keeps the container image small.

Environment variables (Supabase URL, Supabase anonymous key, backend URL, Hugging Face token) are managed through platform-specific secret management — Vercel environment variables for the frontend, Hugging Face Spaces secrets for the backend — and never committed to source control.

---

## CHAPTER 4 — IMPLEMENTATION

### 4.1 AI Inference Engine

#### 4.1.1 From Research Code to Inference Engine

The Sign-IDD model code was originally authored as a research training pipeline: it included data loading, training loops, loss computation, checkpointing, and evaluation logic alongside the model architecture. Adapting it for production inference required a focused extraction and restructuring of the components relevant to the forward inference pass, while keeping the original architecture entirely intact to ensure that the pre-trained weights could be loaded.

The `SignBridgeInference` class (in `sign_bridge_inference.py`) encapsulates the complete inference pipeline. On initialisation, it:

1. Loads the vocabulary from `model_configs/src_vocab.txt`, a pre-built ordered list of all 1,089 DGS gloss tokens from the PHOENIX-2014-T training set. The vocabulary maps each gloss string to an integer index (and vice versa), and includes four special tokens: PAD (padding), BOS (beginning of sequence), EOS (end of sequence), and UNK (unknown).

2. Loads the model configuration from `model_configs/Sign-IDD.yaml`, which specifies all model hyperparameters: encoder depth (6 Transformer layers), embedding dimension (512), number of attention heads (8), feed-forward dimension (2048), dropout rates, diffusion total timesteps (100), and DDIM sampling timesteps (90 by default).

3. Loads the trained checkpoint (`best.ckpt`) from disk into CPU memory using PyTorch's `torch.load`. The checkpoint contains the serialised model weights under the `model_state` dictionary key.

4. Constructs the full model graph using the `build_model()` factory function from the `builders.py` module. This instantiates the Transformer encoder, the ACD diffusion model, and the ACD Denoiser as a single unified nn.Module, and loads the pre-trained weights into the constructed graph.

5. Sets the model to evaluation mode (`model.eval()`) and casts weights to `torch.float32` for CPU inference stability.

The vocabulary reconstruction deserves particular note. The PHOENIX-2014-T vocabulary of 1,089 glosses was not directly available as a standalone file from the original research code; it was embedded in the PHOENIX dataset annotation files. A dedicated script (`build_vocab.py`) was written to process the PHOENIX annotation files, extract all unique gloss tokens in training-set index order, prepend the four special tokens (PAD at index 0, BOS, EOS, UNK), and serialise the result to `src_vocab.txt`. This file is committed to the repository, ensuring that the vocabulary mapping is reproducible without access to the full dataset.

#### 4.1.2 Text-to-Gloss Mapping

The Sign-IDD model operates on sign gloss sequences — standardised notation labels for individual signs — not on raw English text. The PHOENIX-2014-T dataset uses DGS glosses written as capitalised German words (e.g., HEUTE for "today", WETTER for "weather", KALT for "cold"). A translation request arriving as English text must therefore be converted to this gloss notation before being passed to the model.

This conversion is implemented as a two-stage procedure:

In the first stage, a hand-crafted mapping dictionary translates common English temporal and meteorological terms to their DGS gloss equivalents. The dictionary covers approximately 25 high-frequency terms — days of the week, compass directions, common weather conditions, and basic predicates — that together account for a large proportion of PHOENIX sentences.

In the second stage, English words not found in the mapping dictionary are uppercased and looked up directly in the vocabulary index. Many basic DGS glosses happen to coincide with their uppercased German root (or are cognates). Words that fail this lookup are represented by their uppercased form, effectively routing them through the unknown token mechanism.

The resulting gloss list is wrapped with BOS and EOS tokens and converted to integer indices for input to the model. The total length of the animation to generate is estimated dynamically: the model produces approximately 15 frames per input gloss, plus a 20-frame buffer for the sequence start and end transitions, with a floor of 60 frames to ensure minimal animation length.

This approach to text-to-gloss mapping is an acknowledged simplification. A linguistically rigorous system would use a statistical or neural text-to-gloss translation model trained on parallel text-gloss corpora. The current hand-crafted mapping serves the proof-of-concept scope of the project while leaving a clear path for improvement.

#### 4.1.3 The DDIM Sampling Process

The core of the inference engine is the DDIM sampling loop implemented in `ACD.ddim_sample()`. This method implements the reverse diffusion process that transforms a random noise tensor into a structured sign language pose sequence.

The method begins by determining the sequence of timesteps to traverse. For S sampling steps over a total of T diffusion timesteps, a uniformly spaced sequence of S+1 values is drawn from [-1, T-1], reversed to count downward from T-1 to -1, and converted to adjacent pairs: (T-1, t_(S-1)), (t_(S-1), t_(S-2)), ..., (t_1, -1). Each pair (t, t_next) defines one denoising step.

An initial noise tensor of shape (1, T_frames, 150) is drawn from a standard Gaussian distribution on the compute device, representing a completely unstructured configuration of 50 skeletal joints over T_frames frames.

At each timestep pair (t, t_next), the following operations occur:

The current noise tensor is passed to `model_predictions()`, which first applies the iconicity (ID) expansion to each frame, transforming the 150-dimensional XYZ representation into a 350-dimensional geometrically enriched feature. The ACD Denoiser then takes this expanded noisy pose, combined with the encoder output of the input glosses and a time-step embedding, and produces a predicted clean pose estimate. From this clean estimate, the corresponding predicted noise is back-calculated using the DDIM noise prediction formula, mixing the body and hand beta schedule parameters per channel.

The update formula then computes the next-step sample by combining: the clean pose prediction scaled by the per-channel square root of the next-step alpha_cumprod; the predicted noise scaled by a deterministic direction coefficient; and optionally a small amount of fresh Gaussian noise scaled by per-channel sigma. This stochastic element (controlled by eta = 1.0) is what causes different runs with different random seeds to produce different outputs — the generative variability of the model.

After all S steps, the collection of clean pose estimates from each step is returned. The final estimate — from the last denoising step — is used as the generated pose sequence.

The number of sampling steps S is a runtime parameter. The Standard inference mode uses S = 20, which produces good-quality output in reasonable time. The High-Fidelity mode uses S = 60, traversing a finer-grained denoising trajectory and producing smoother, more detailed motion at the cost of longer inference time.

#### 4.1.4 Video Rendering

The output of the DDIM sampling is a Python array of shape [T_frames][50][3] — T_frames snapshots of 50 joint positions in 3D space. To make this human-interpretable, the raw skeletal data is rendered into an MP4 video.

The rendering is performed by `video_renderer.py`, which iterates over the frame sequence and for each frame:

- Creates a 3D Matplotlib figure.
- Plots the 50 joint positions as a scatter plot in the XZ plane (the frontal view of the signer).
- Connects joints with line segments according to the human kinematic hierarchy — a predefined list of parent-child joint index pairs that encode the anatomical connectivity of the human body (e.g., elbow is a child of shoulder, wrist is a child of elbow).
- Saves the figure as a PNG image frame.

Once all frames are rendered, FFmpeg is invoked as a subprocess to assemble the frame sequence into an MP4 video at 25 fps. The output video is saved to the backend's static file directory with a unique UUID-derived filename, preventing filename collisions between concurrent requests.

### 4.2 Backend API Service

#### 4.2.1 API Design

The primary AI backend exposes a minimal, focused API designed around the needs of the frontend:

`POST /translate` accepts a JSON body containing the user's input text and returns a JSON response with the URL of the generated MP4 video and the processed text. This is the endpoint called in Standard mode.

`POST /translate_hq` is functionally identical but routes the request to a separately loaded High-Fidelity model instance that uses the full 1.1 GB weight checkpoint (when available) and performs 60 DDIM sampling steps.

`GET /static/{filename}` serves the generated MP4 files as HTTP responses with the `video/mp4` media type, allowing the frontend to embed them directly in an HTML video element.

The secondary `sign-idd-api` service exposes:

`GET /search?q={query}` performs fuzzy substring matching of the query against an in-memory index of pre-generated video filenames, returning a ranked list of matching video names and their corresponding scores.

`GET /video/{name}` streams the content of a named pre-generated video file as a `FileResponse`.

`GET /videos` lists all available pre-generated videos for browsing.

All endpoints include CORS headers permitting cross-origin requests from the Vercel frontend domain (and all origins during development), and return appropriate HTTP error codes (404 for missing resources, 500 with descriptive messages for inference failures).

#### 4.2.2 Model Loading Strategy

The Sign-IDD model, even after compression, requires loading a 442 MB checkpoint file and initialising a Transformer with over 50 million parameters. This initialisation takes between 30 and 90 seconds on a CPU-only server. If model loading blocked the HTTP server startup, every incoming request during the loading period would fail, and the first request after every cold start would time out.

To decouple model availability from server responsiveness, the `SignModel` class implements the **Singleton + Asynchronous Background Loading** pattern. The class uses Python's `threading.Lock` to ensure that exactly one instance is ever created (the Singleton guarantee). Model loading is performed in a **daemon thread** started when the FastAPI application starts via its `lifespan` context manager.

Inference endpoints check the `is_loaded` flag. If the model is still loading, the endpoint immediately returns a 500 response with the message "Model is still loading — please retry in 30 seconds." If loading has failed (due to a missing checkpoint or download failure), the endpoint returns the stored error message. Once loading succeeds, all subsequent requests proceed to inference directly.

This pattern means the HTTP server is always immediately responsive. Users who access the service during the cold-start window receive a clear, actionable response and can retry; they never experience an unresponsive server.

#### 4.2.3 Request Handling and Concurrency

FastAPI's asynchronous request handling means the server can accept new connections while an existing inference request is being processed. Since PyTorch CPU inference is CPU-bound (not I/O-bound), it runs synchronously within a thread pool. The Python Global Interpreter Lock (GIL) means that only one thread runs Python code at a time, but I/O operations (file reading, video writing) release the GIL, allowing some effective concurrency.

Each inference request generates a video file with a unique UUID-derived filename (`gen_{uuid4().hex[:8]}.mp4`), ensuring that concurrent requests do not overwrite each other's outputs. The output directory is created on startup if absent.

### 4.3 Model Compression and Cloud Deployment

#### 4.3.1 Why Compression Was Necessary

A PyTorch model checkpoint produced during training contains more than just the model weights. It also stores the complete state of the optimiser — in this case, Adam — which includes the first moment estimate (exponential moving average of gradients) and the second moment estimate (exponential moving average of squared gradients) for every trainable parameter. These are essential during training for adaptive learning rate computation but are entirely unused during inference. For a model with ~50 million parameters, the Adam state contributes approximately 3× the raw parameter size in additional storage.

The full checkpoint was 1.14 GB. Hugging Face Spaces free tier imposes practical storage and memory limits; similarly, committing files larger than 100 MB to Git requires Git LFS, adding friction to repository management. The compression goal was to produce a checkpoint small enough to be practically deployable on free-tier infrastructure.

#### 4.3.2 The Compression Pipeline

A dedicated script (`shrink_model.py`) implements the compression in two sequential steps:

**Step 1 — Optimiser State Stripping**: The full checkpoint is loaded into CPU memory. The `model_state` dictionary — containing only the model parameters as serialised tensors — is extracted. All other keys in the checkpoint (the optimiser state, step counters, epoch metadata) are discarded. The stripped checkpoint is re-serialised preserving only `model_state` and two lightweight metadata values (`steps`, `total_tokens`). This step alone removes approximately 300 MB.

**Step 2 — FP16 Quantisation**: The remaining model parameter tensors, stored as 32-bit floats (`torch.float32`, 4 bytes per value), are converted to 16-bit half-precision (`torch.float16`, 2 bytes per value). This halves the storage requirement of the parameter tensors, removing a further ~400 MB. FP16 precision is sufficient for inference — the numerical error introduced by the lower precision is below the perceptual threshold for the output motion quality.

After loading the compressed checkpoint at inference time, the weights are immediately cast back to `torch.float32` before any forward pass computations, since CPU tensors in PyTorch perform mixed-precision accumulation more reliably in full precision.

The combined effect: **1.14 GB → 442 MB**, a 61.2% reduction, achieved through software-only transformations applied post-training.

#### 4.3.3 Automated Weight Sourcing

Because the compressed weights (442 MB) exceed Git's recommended single-file limit and would make repository cloning impractically slow, they are not committed to the application repository. Instead, they are hosted in a dedicated private Hugging Face Model Hub repository (`ExploWebsite/SignBridge-Weights`).

The backend implements an automated sourcing routine within the asynchronous loading thread: it checks whether the expected weight file exists in the persistent storage path. If the file is absent (as it would be on a fresh container deployment), the `huggingface_hub.hf_hub_download()` function is called with the Hub repository ID, the filename, and an authentication token sourced from the `HF_TOKEN` environment secret. The downloaded file is placed in the correct location and renamed to the expected filename, after which normal model initialisation proceeds.

This design means that a fresh deployment of the backend container — with no locally stored weights — becomes fully operational automatically within a few minutes of startup, without any manual intervention.

### 4.4 Frontend Web Application

#### 4.4.1 Application Architecture

The frontend is built using the **Next.js 14 App Router**, which organises the application into route segments corresponding to page directories. The application routes are:

`/` — The landing page. This is the user's first impression of SignBridge. It presents a hero section with the platform's value proposition, a features section (Text to Sign, AI-Powered Generation, Sign to Text), a mission statement section, and summary statistics. The page uses animated entrance effects and a generative floating-line background to communicate the platform's AI-forward identity.

`/text-to-sign` — The primary product interface. This is where users interact with the translation system.

`/sign-to-text` — A Beta module page for the inverse task of recognising sign language from camera input.

`/login` — The authentication entry point, providing Google OAuth sign-in via Supabase Auth.

`/auth/callback` — The server-side route handler that processes the OAuth redirect from Google, exchanges the authorisation code for a session token, and redirects the authenticated user back to the application.

The global layout wraps all pages with a shared `Header` component — a fixed, always-visible navigation bar featuring the SignBridge logo, a floating pill-shaped navigation menu centred on the page, and right-aligned authentication controls.

#### 4.4.2 The Text-to-Sign Translation Interface

The translation interface (`TextToSignClient.tsx`) is the centrepiece of the frontend. It is structured as a horizontal split-panel layout occupying the full viewport height beneath the header:

**Left Panel — Input and Control**: This panel contains the text input field, mode selector, and results list.

The mode selector is a three-state toggle that determines how the input is processed:

- *Search Videos* mode submits the query to the `sign-idd-api` `/search` endpoint and displays a list of matching pre-generated video names. The user selects a result to play it in the right panel. This mode is designed for fast, dictionary-style lookup of known signs.

- *Live AI Bridge* mode (Standard) submits the input text to the backend `/translate` endpoint. The frontend enters a loading state displaying "Diffusion sampling in progress..." while the server generates the animation. When the response arrives, the video URL is rendered in the right panel.

- *AI Bridge HQ* mode routes the request to the `/translate_hq` endpoint, generating higher-quality output at the cost of longer generation time. This mode is visually distinguished with an amber colour scheme to indicate premium quality.

The input field responds to the Enter key, the form state is carefully managed to handle concurrent loading states across the three modes, and error messages are displayed inline with actionable context (indicating which service is unavailable).

**Right Panel — Animation Display**: This panel renders the translation output. It supports three rendering modes:

If the backend returns a `video_url`, an HTML5 `<video>` element is rendered, configured for autoplay-on-load, inline playback, and loop. The source URL points to the static video file on the backend server.

If the backend returns raw `skeletons` data (the 3D joint array, returned when video rendering is bypassed), the `SkeletonViewer` component renders a Three.js-based 3D visualisation of the skeletal animation in real-time within the browser.

When no video or skeleton data is present, the panel shows a contextual placeholder describing what the user should do next, customised for each of the three modes.

Below the video player (when a video is displayed), an action strip provides two user interaction features: a **Save to Favourites** button that writes the video name and user ID to the Supabase `favorites` table (requiring authentication), and **thumbs-up / thumbs-down feedback buttons** that write to the `feedback` table, enabling future quality assessment of translations.

#### 4.4.3 Three-Dimensional Skeleton Viewer

The `SkeletonViewer` component provides a browser-based, real-time 3D visualisation of raw skeletal data without the need for a pre-rendered video file. It is implemented using Three.js (via React Three Fiber), rendering:

- The 50 skeletal joints as coloured sphere meshes placed at their XYZ coordinates.
- Bone connections as cylindrical meshes stretched between parent and child joint positions, following the predefined human kinematic hierarchy.
- An animated playback loop that advances through the frame sequence at a configurable frame rate.

This component serves two purposes: it allows users to see the raw AI output without the video rendering latency, and it provides a more interactive visualisation that can be rotated and inspected.

#### 4.4.4 Interface Design

The visual design of SignBridge reflects the positioning of the platform as a premium, AI-forward accessibility tool. The design language is built on several coherent principles:

**Dark-mode-first**: The entire interface uses a near-black (`#000000`) background. This reduces visual fatigue for extended use, creates high contrast for the animated content, and aligns with the aesthetic conventions of AI and developer-facing products.

**Gradient Identity**: A consistent indigo-violet-pink gradient (`#6366f1` → `#a855f7` → `#831843`) is used as the primary accent across typography, buttons, and interactive elements. This gradient serves as SignBridge's visual signature.

**Animated Background**: The landing page features a `FloatingLines` component that renders generative multi-wave line patterns that respond to mouse movement, creating an impression of a living, AI-generated visual environment.

**Micro-interactions**: `ScrollReveal` components apply staggered entrance animations to content sections as they enter the viewport. `TiltedCard` components apply a 3D perspective tilt in response to cursor position on feature cards. These interactions provide tactile feedback and signal interactivity without distracting from content.

**Pill Navigation**: The navigation bar is a floating, glassmorphism-styled capsule centred horizontally at the top of the viewport, detached from the page edges. This provides clear, accessible navigation while reinforcing the design's modern aesthetic.

### 4.5 Database Design and User Ecosystem

#### 4.5.1 Authentication System

User authentication is handled entirely by **Supabase Auth**, using Google as the OAuth provider. The authentication flow works as follows: when a user clicks "Log in," they are redirected to Google's OAuth consent screen. After consent, Google redirects to the application's `/auth/callback` route with an authorisation code. The Next.js route handler exchanges this code for a session token using Supabase's server-side client and sets the session in a secure HTTP-only cookie. The frontend reads session state via the Supabase JavaScript client's `onAuthStateChange` event subscription, updating the Header component to display the user's avatar and a sign-out button.

Row-Level Security (RLS) policies in Supabase restrict data access: users can only read their own `favorites` and `search_history` records, while writes to `search_history` and `feedback` are permitted for both authenticated and anonymous users (the latter stored with a null `user_id`).

#### 4.5.2 Database Schema

Three application tables support the user ecosystem, each designed for a specific interaction type:

**`search_history`** serves as a usage journal — every search query and generation request is logged with the query string, the user's ID (if authenticated), and a timestamp. This data supports personalised history display, usage analytics, and future recommendation features.

**`favorites`** allows authenticated users to bookmark videos they find useful or accurate. Each record stores the video filename, the user ID, and a creation timestamp. A unique constraint on (user_id, video_name) prevents duplicate bookmarks. The video filename is sufficient to reconstruct the full video URL at display time.

**`feedback`** captures explicit quality signals from users. Each record stores the video filename, the user ID (nullable for anonymous feedback), a boolean `is_positive` flag, and a timestamp. The aggregate of positive and negative feedback over each video provides a dataset for future model evaluation and fine-tuning.

---

## CHAPTER 5 — RESULTS AND DISCUSSION

### 5.1 Functional Outcomes

The following capabilities were implemented and verified as correctly operational through end-to-end testing:

**Text-to-Sign Generation (Standard)**: English text input is correctly processed through the gloss mapping, Transformer encoding, 20-step DDIM sampling, video rendering, and HTTP delivery pipeline. The frontend successfully displays the generated animation in the video player.

**Text-to-Sign Generation (High-Fidelity)**: The HQ pathway using 60 DDIM sampling steps functions correctly, producing longer-running but higher-quality output compared to the Standard mode.

**Pre-Generated Video Search**: The `sign-idd-api` correctly indexes available video files on startup and returns relevant fuzzy-matched results for topic queries such as "tagesschau," "heute," and "REGEN."

**Three-Dimensional Skeleton Viewer**: Raw skeletal data returned by the API (when video rendering is bypassed) is correctly rendered as an interactive Three.js 3D animation in the browser, with correct kinematic connections and animated playback.

**User Authentication**: Google OAuth sign-in and sign-out function correctly, with session state persisted across page navigations and browser refreshes.

**User Data Features**: Authenticated users can save videos to Favourites, with the save operation reflected in the Supabase `favorites` table. Thumbs-up and thumbs-down feedback correctly writes records to the `feedback` table with the correct user ID and video name.

**Anonymous Usage**: Anonymous users can search and generate translations; their queries are recorded in `search_history` with a null `user_id`.

**Input-Dependent Output**: Generated animations correctly reflect the length and composition of the input: longer input gloss sequences produce longer animations, and the generated joint trajectories differ between different inputs, confirming that the model is producing input-conditioned rather than fixed or random output.

### 5.2 Model Compression Results

The model compression pipeline achieved the following outcomes:

| Metric | Value |
|:---|---:|
| Original trained checkpoint size | 1,140 MB |
| Size after optimiser state stripping | ~740 MB |
| Size after FP16 quantisation | 442 MB |
| Total reduction | 61.2% |
| Deployment target | Hugging Face Spaces (free tier) |
| Inference precision | float32 (restored at load time) |

The compressed model loads successfully, generates correctly structured pose sequences, and produces animations that are qualitatively indistinguishable from full-precision inference on the same inputs, confirming that the compression introduced no perceivable quality degradation.

### 5.3 Deployment Status

Both application tiers are successfully deployed and publicly accessible:

The **frontend** is deployed on Vercel and served via HTTPS with automatic CDN distribution. Continuous deployment is configured: every push to the main branch of the GitHub repository triggers a new Vercel build and deployment within approximately 90 seconds.

The **AI backend** is deployed on Hugging Face Spaces as a Docker container. On cold start, the automated weight download mechanism downloads the compressed checkpoint from the Hugging Face Model Hub and initialises the model. The Uvicorn server is accessible via the public Hugging Face Spaces HTTPS URL, and the frontend is configured with this URL as `NEXT_PUBLIC_BACKEND_URL`.

### 5.4 Discussion of Output Quality

The generated sign language animations are functional and input-dependent, but their quality should be assessed honestly within the context of the project scope.

**Strengths**: The animations correctly reflect input length — a three-word input produces a shorter animation than a six-word input, consistent with the dynamic sequence length estimation. The spatial trajectories of the generated joints vary across inputs, confirming that the model is producing genuinely conditioned output and not degenerating into a fixed average pose. The video rendering pipeline produces clean, human-readable skeletal animations that clearly visualise the sign motion.

**Limitations in Output Naturalness**: The generated motions tend toward a conservative range of joint displacement compared to natural, expressive signing. This is an expected characteristic of the PHOENIX-2014-T dataset, which captures the relatively constrained, formal signing style of television broadcast signers rather than conversational signing. Additionally, the project uses a model checkpoint that may not represent the fully converged end of training; further training epochs would likely produce higher-variance, more natural motion.

**Vocabulary Coverage**: The text-to-gloss mapping covers approximately 25 explicitly handled English-to-DGS gloss translations. Input words outside this mapping are passed through the vocabulary lookup, which may succeed for many common terms but fails for vocabulary not in the PHOENIX-2014-T lexicon. A neural text-to-gloss translation model would substantially expand coverage and handle paraphrase and morphological variation.

**Coarticulation**: The generated sequence represents the sign sentence as a whole temporal trajectory, not as a concatenation of individual sign clips. However, the model does not explicitly model coarticulation — the smooth, phonologically motivated blending of adjacent signs. More training and a richer dataset would improve this aspect.

---

## CHAPTER 6 — CONCLUSION AND FUTURE WORK

### 6.1 Summary and Learnings

This project set out to answer a specific question: can a state-of-the-art sign language production AI model be made genuinely accessible through a web platform, deployable within free-tier cloud infrastructure, usable by anyone with a browser? The answer, demonstrated through SignBridge, is yes — with deliberate engineering.

The project traversed the full distance from research to deployment. It began with a deep study of the Sign-IDD architecture — understanding the DDIM sampling process, the multi-rate noise scheduling, the PINN loss formulation, and the ACD Denoiser mechanics — and then adapting the training-focused research code into a production inference engine. It continued through API design, model compression, cloud containerisation, and the construction of a full-featured, user-facing web application.

The key learnings from this project can be grouped into three areas:

**Understanding Generative AI Systems**: Working directly with a diffusion model for motion generation provided deep practical understanding of the DDIM framework — how the choice of sampling steps, noise schedules, and the eta parameter affect output quality and diversity. The multi-rate scheduling design in Sign-IDD was particularly instructive as an example of domain-specific adaptation of a general framework.

**The Gap Between Research and Production**: Research ML systems are designed for training reproducibility and experimental flexibility. Production inference systems require robustness, predictable latency, size efficiency, and device portability. Bridging this gap — through checkpoint compression, asynchronous loading patterns, and device-agnostic tensor operations — was one of the most substantive engineering challenges of the project and a lesson with broad applicability.

**Full-Stack System Design**: The project required integrating components across fundamentally different technology stacks — Python/PyTorch for AI, Python/FastAPI for the API layer, TypeScript/React/Next.js for the frontend, Docker for containerisation, and Supabase for data. Designing clean interfaces between these layers — particularly the JSON API contract between backend and frontend — ensured that each tier could be developed, tested, and deployed independently.

Beyond the technical dimension, the project provided direct engagement with the question of AI for accessibility. The constraints of the Deaf community's digital experience — their dependence on visual media, the inadequacy of text-only interfaces, the absence of quality sign language tools — are not abstract. Building even a proof-of-concept sign language generation system makes these constraints concrete and motivates the future work described below.

### 6.2 Future Directions

**Linguistic Coverage and Sign Language Generalisation**: The most immediate limitation of SignBridge is its restriction to DGS and the PHOENIX weather domain. Expanding to broader sign language coverage would require training or fine-tuning on larger, more diverse corpora. For German Sign Language, the Public DGS Corpus (Konrad et al., 2020) offers a substantially richer and more varied dataset. For other sign languages, datasets such as BOBSL (British Sign Language) and ASL-Citizen (American Sign Language) are available for research use.

**Neural Text-to-Gloss Translation**: Replacing the hand-crafted English-to-gloss mapping with a neural translation model — a sequence-to-sequence Transformer or a fine-tuned language model — would dramatically expand the input vocabulary and handle natural language variation, paraphrase, and morphological inflection that the current dictionary cannot manage.

**Real-Time Sign-to-Text Recognition**: The Sign-to-Text pathway is currently a placeholder interface. Implementing it would require: extracting signer pose from a webcam video stream (achievable with MediaPipe Holistic in real time); classifying or sequentially decoding the pose stream into gloss labels using a trained temporal model; and optionally translating the resulting gloss sequence into fluent text. This would complete the bidirectional communication bridge that is SignBridge's stated goal.

**Inference Latency Reduction**: Current inference latency (30–90 seconds) is acceptable for a demonstration but not for conversational use. Potential approaches include: model distillation (training a smaller, faster student model to imitate the larger teacher); exporting the model to ONNX and using optimised runtime inference; or deploying on GPU-accelerated serverless infrastructure when cost constraints allow.

**Coarticulation-Aware Generation**: A more sophisticated motion generation model would explicitly model the coarticulation between adjacent signs — the phonologically motivated modification of a sign's shape and trajectory based on what comes before and after it. This is an active research problem and would require training data annotated at the sub-gloss phonological level.

**Feedback-Driven Improvement**: The user feedback system (thumbs-up / thumbs-down per video) was designed not only as a UX feature but as a data collection mechanism for future model improvement. With sufficient feedback data, a Reinforcement Learning from Human Feedback (RLHF) approach could be used to fine-tune the generation model toward outputs that users rate positively.

---

## REFERENCES

1. **Ho, J., Jain, A., and Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models.* Advances in Neural Information Processing Systems (NeurIPS), Vol. 33, pp. 6840–6851.

2. **Song, J., Meng, C., and Ermon, S.** (2021). *Denoising Diffusion Implicit Models.* International Conference on Learning Representations (ICLR 2021). arXiv preprint arXiv:2010.02502.

3. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I.** (2017). *Attention is All You Need.* Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.

4. **Camgoz, N. C., Hadfield, S., Koller, O., Ney, H., and Bowden, R.** (2018). *Neural Sign Language Translation.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7784–7793.

5. **Camgoz, N. C., Koller, O., Hadfield, S., and Bowden, R.** (2020). *Sign Language Transformers: Joint End-to-End Sign Language Recognition and Translation.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

6. **Koller, O., Forster, J., and Ney, H.** (2015). *Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers.* Computer Vision and Image Understanding (CVIU), Vol. 141, pp. 108–125.

7. **Raissi, M., Perdikaris, P., and Karniadakis, G. E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, Vol. 378, pp. 686–707.

8. **Zelinka, J. and Kanis, J.** (2020). *Neural Sign Language Synthesis: Words Are Our Glosses.* Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV).

9. **RWTH-PHOENIX-Weather 2014T Dataset.** Institut für Sprach- und Kommunikationstechnik, RWTH Aachen University. Available at: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

10. **FastAPI.** Sebastián Ramírez. High performance, easy to learn, fast to code, ready for production Python web framework. Available at: https://fastapi.tiangolo.com/

11. **Next.js.** Vercel Inc. The React Framework for the Web. Available at: https://nextjs.org/

12. **Supabase.** Supabase Inc. The Open Source Firebase Alternative. Available at: https://supabase.com/

13. **PyTorch.** Meta AI. An open source machine learning framework. Available at: https://pytorch.org/

14. **Hugging Face Hub.** Hugging Face Inc. The AI community building the future. Available at: https://huggingface.co/

---

*End of Report*

---

**Harshit Vaghamshi** | Roll No. 24075091 | B.Tech | IIT (BHU), Varanasi | 19 April 2026
