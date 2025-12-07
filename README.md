# GenAI_FinalProject

# Generating Product Images from Customer Reviews

Final project for **94-844 Generative AI Lab (Fall 2025)**  
Team: **Maithreyee Sampath, Piyush Kautkar, Shane Judge**

This project explores how a **language model** can read product descriptions and customer reviews, extract meaningful **visual and functional cues**, and then drive **diffusion models** to generate realistic product images.

We:

- Selected **three Amazon products** from different categories  
- Collected **product descriptions and 30 customer reviews**  
- Used an **OpenAI LLM** to build structured product profiles (JSON)  
- Used those JSON profiles to design prompts for **two diffusion models** on Hugging Face:
  - `stabilityai/sd-turbo`
  - `stabilityai/sdxl-turbo`
- Ran diffusion models on **Google Colab** (T4 GPU) due to local hardware limits  

The generated images capture the product category, approximate shape, and mood. They do not reproduce exact branding or tiny text. Across all products, **sdxl-turbo** produced more coherent and visually appealing images than **sd-turbo**.

---

## 1. Repository Structure

```text
GenAI_FinalProject/
  notebooks/
    01_environment_setup.ipynb    # Environment checks, key loading
    02_data_collection.ipynb      # Product metadata, descriptions, reviews
    03_review_analysis.ipynb      # LLM-based review analysis, JSON outputs
  (Q3 image generation is implemented in a linked Colab notebook; see Section 4.)

  data/
    metadata/
      product_list.csv            # Three products: id, name, category, marketplace, URL
    raw/
      product1_description.txt
      product2_description.txt
      product3_description.txt
      product1_reviews.csv
      product2_reviews.csv
      product3_reviews.csv
    processed/
      product1_analysis.json
      product2_analysis.json
      product3_analysis.json
    images/
      stable_diffusion/
        sd-turbo/
          P1/  # clean, detail, in_use images for product 1
          P2/  # clean, detail, in_use images for product 2
          P3/  # clean, detail, in_use images for product 3
        sdxl-turbo/
          P1/
          P2/
          P3/
      # openai/                     # Not used – OpenAI image model access was restricted
    image_generation_log.csv       # Log of all generated images and prompts

  .env                             # Local env vars (not committed)
  requirements.txt                 # Python dependencies
  .gitignore                       # Ignore venv, raw data, checkpoints, system files
  README.md                        # This file
  ```
## 2. Setup Instructions (Local Environment)

All notebooks for **Q1 (data)**, **Q2 (LLM review analysis)**, and **Q3 (prompt construction + image generation pipeline)** live in this repository under `notebooks/` and can be run locally if you have sufficient GPU resources.

In our workflow:
- Q1 and Q2 were run locally.
- Q3 logic (prompt building and pipeline code) is in the repo, but the **actual image generation with diffusion models** was executed on **Google Colab with a T4 GPU** for practicality. The Colab workflow mirrors the same code structure.

### 2.1. Clone the repository and activate virtual environment

```bash
git clone https://github.com/maithreyee-sampath/GenAI_FinalProject.git
cd GenAI_FinalProject
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows (if needed)
```
### 2.2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 2.3. Configure API keys
Create a .env file in the project root:
```bash
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
```
OPENAI_API_KEY is required for Q2 LLM analysis and the agentic workflow.
HF_TOKEN is needed if you want to reproduce the diffusion models (locally or in Colab).

## 3. Q1 – Product Selection and Data Collection

**Notebook:** `notebooks/02_data_collection.ipynb`

This project uses three Amazon products from different categories:

1. **Sony WH-1000XM5 – Wireless noise-cancelling headphones**  
   - Category: Electronics  

2. **Ninja AF101 – 4-quart air fryer**  
   - Category: Kitchen appliances  

3. **CeraVe Hydrating Facial Cleanser (16 oz)**  
   - Category: Beauty / Skincare  


### 3.1. What this notebook does

The notebook:

1. **Creates a metadata table** 

   ```text
   data/metadata/product_list.csv

  with columns:
  - Product_ID (P1, P2, P3)
  - Product_Name
  - Category and Item
  - Marketplace (Amazon)
  -URL
  
   
2. **Stores full product descriptions as plain text:** 

   ```text
   data/raw/product1_description.txt
   data/raw/product2_description.txt
   data/raw/product3_description.txt

3. **Stores ten manually selected customer reviews per product:**
```text
data/raw/product1_reviews.csv
data/raw/product2_reviews.csv
data/raw/product3_reviews.csv

Each CSV contains:

- One row per review
- Rating
- Review text
```
### 3.2. Design considerations

Products were chosen to maximise diversity in:

- Physical form (headphones vs appliance vs bottle)

- Usage context (travel, cooking, skincare routine)

- Visual characteristics (shape, materials, label design)

Ten reviews per product give enough variety in:

- Sentiment (mix of positive and negative)

- Topics (features, pain points, usage patterns)

Storing data as simple .txt and .csv files keeps the pipeline:

- Easy to inspect

- Easy to reuse in other notebooks (Q2, Q3, Q4)

## 4. Q2 – LLM-Based Review Analysis

**Notebook:** `notebooks/03_review_analysis.ipynb`

### 4.1. Goal

Convert unstructured customer reviews into a **structured JSON representation** that is directly useful for image prompt design in Q3 and for the agentic workflow in Q4.

### 4.2. LLM and prompting strategy

We use `gpt-4o-mini` via the OpenAI API.

For each product:

1. Load the ten reviews from:

   ```text
   data/raw/product1_reviews.csv
   data/raw/product2_reviews.csv
   data/raw/product3_reviews.csv
   ```

2. Concatenate them into a single text block with a clear separator between reviews  
   (for example: `\n--- REVIEW ---\n`).

3. Call the OpenAI API with:

   - **A system prompt** explaining that:
     - The input is a set of Amazon reviews for a specific product.
     - The model must output only valid JSON.

   - **A user prompt** that requests this schema:

     ```json
     {
       "summary": "",
       "key_benefits": [],
       "pain_points": [],
       "visual_attributes": [],
       "usage_contexts": [],
       "tone_words": []
     }
     ```

4. Parse the JSON and save to:

   ```text
   data/processed/product1_analysis.json
   data/processed/product2_analysis.json
   data/processed/product3_analysis.json
   ```

These files act as compact product profiles summarising what customers say about each item.

### 4.3. Types of analysis covered

The JSON schema lets us cover several analysis dimensions:

- **Text summarisation**  
  - `summary` captures the overall story of the reviews.

- **Feature and visual cue extraction**  
  - `key_benefits` and `pain_points` capture functional pros and cons.  
  - `visual_attributes` captures appearance: shape, colour, materials, packaging, design.

- **Topics and usage patterns**  
  - `usage_contexts` captures situations and scenarios, such as:
    - commuting and flights for the headphones  
    - weeknight dinners for the air fryer  
    - morning and evening routines for the cleanser

- **Sentiment-like signals**  
  - `tone_words` collects emotional and tonal descriptors such as:
    - “premium”, “comfortable”, “noisy”, “gentle”, “soothing”, “frustrating”.

### 4.4. Prompt strategies and design choices

We considered two main strategies:

1. **Simple zero-shot summary / pros-cons prompt**

   - Ask the model to summarise the reviews and list pros and cons in free text.  
   - Good for quick understanding but harder to reuse programmatically.

2. **Structured JSON-only prompt (final choice)**

   - Enforce the schema above and require JSON-only output.  
   - Much more reliable for downstream code that:
     - builds prompts for diffusion models, and  
     - feeds into the agentic workflow in Q4.

The final pipeline uses the **structured JSON** approach, trading some flexibility for better automation and reproducibility.

### 4.5. Chunking, RAG, and vector database considerations

The combination of product description plus ten reviews per product was well within the context window of `gpt-4o-mini`.

Because of this:

- No advanced chunking strategy was required.  
- No vector database or full RAG pipeline was necessary.

However:

- We still separate reviews with delimiters before sending them to the model.  

This makes it straightforward to extend the system later by:

- turning each review into an embedding,  
- storing them in a vector database, and  
- implementing RAG-style retrieval over reviews.

### 4.6. Connection to Q3 and Q4

The output of Q2 is intentionally **prompt-ready** and **agent-ready**:

- `visual_attributes` and `usage_contexts` feed directly into text prompts for diffusion models in Q3.
- The full JSON profile is consumed by the agentic workflow in Q4 to reason about product features, user needs, and visual directions.

## 5. Q3 – Image Generation with Diffusion Models

**Notebook logic:** `notebooks/03_review_analysis.ipynb` for prompt prep  
**Image generation execution:** Google Colab notebook (T4 GPU) linked in this README

### 5.1. Goal

Use the structured analyses from Q2 to:

- Craft prompts that reflect what customers describe.
- Generate 3–5 images per product using two different diffusion models.
- Compare generated images with real Amazon listing photos and analyse differences.

### 5.2. Model selection and environment constraints

**Initial plan:**

- Use an OpenAI image model such as `gpt-image-1`.
- Use full Stable Diffusion XL locally via Hugging Face.

**Constraints we hit:**

- OpenAI image calls returned an organisation verification error that we could not resolve as students.
- Running full SDXL on a MacBook with MPS caused repeated out-of-memory issues and black outputs.

**Final design:**

- Keep the prompt construction logic in the repo (aligned with Q2 outputs).
- Run image generation on Google Colab with a T4 GPU.
- Use two fast diffusion models from Hugging Face:
  - `stabilityai/sd-turbo`
  - `stabilityai/sdxl-turbo`

**Reasons for this choice:**

- Both are optimised for fast text-to-image generation on modest GPUs.
- `sd-turbo` is lightweight and great for quick exploration.
- `sdxl-turbo` is distilled from SDXL and produces higher quality, more coherent images.
- Both are well documented and widely used, which supports reproducibility.

### 5.3. Prompt construction from JSON

For each product, we build prompts for three views:

- `clean` – studio product shot
- `detail` – close-up or emphasis shot
- `in_use` – lifestyle context shot

**General template for clean view:**

> High quality studio product photograph of [product name].  
> Shows [visual attributes].  
> Neutral softly lit background.  
> Focus on the product, high detail, no text overlay.

**General template for in_use view:**

> Realistic lifestyle photo of [product name] used in [usage contexts].  
> Shows [relevant benefits and visual attributes].  
> Natural lighting, candid feel, high detail, no text overlay.

**Detail views** reuse the clean prompt and add close-up or context emphasis, such as:

- ingredients and texture for the cleanser, or
- steam and crispness for the air fryer.

Bracketed segments are filled using phrases from:

- `visual_attributes`
- `usage_contexts`
- `tone_words`

> **Note:** The CLIP encoder warns when prompts exceed 77 tokens, so some trailing words (such as “no text overlay”) are truncated.  
> Because we place key descriptive content early, truncation does not break core meaning.

### 5.4. Image generation protocol

For each product:

1. Generate clean, detail, and in_use images with `sd-turbo`
2. Generate clean, detail, and in_use images with `sdxl-turbo`

This yields:

> 3 prompts × 2 models × 3 products = **18 images**,  
> satisfying the requirements for “3–5 images per product” and “two models”.

**Images are saved under:**

```
data/images/stable_diffusion/sd-turbo/P1
data/images/stable_diffusion/sd-turbo/P2
data/images/stable_diffusion/sd-turbo/P3

data/images/stable_diffusion/sdxl-turbo/P1
data/images/stable_diffusion/sdxl-turbo/P2
data/images/stable_diffusion/sdxl-turbo/P3
```

Every generation is logged in:

```
data/image_generation_log.csv
```

with:

- product_id
- view
- model
- prompt
- file_path
- timestamp

### 5.5. Qualitative comparison with actual product images

We compare generated images to Amazon listing photos:

**Sony WH-1000XM5**

- Both models capture the over-ear headphone form and premium feel.
- `sd-turbo` is noisier and less faithful in proportions.
- `sdxl-turbo` produces more realistic lighting and shapes but cannot reproduce exact logos.

**Ninja AF101 Air Fryer**

- Both models produce convincing air fryer shapes with crisp food and kitchen visuals.
- `sd-turbo` is more stylised and high contrast.
- `sdxl-turbo` creates clearer geometry, props, and commercial-style lighting.

**CeraVe Hydrating Cleanser**

- `sd-turbo` generates generic skincare bottles with blue labels.
- `sdxl-turbo` produces tall white pump bottles, blue/green labels, plants, and citrus—close to CeraVe branding without exact typography.

**Across all products:**

- `sd-turbo` is useful for rapid exploration but visually noisy.
- `sdxl-turbo` yields smoother textures, better lighting, and stronger alignment with JSON-driven prompts.
- Neither model reproduces logos or small text, but both capture category and brand mood well.

### 5.6. Colab execution

Because local GPU resources were limited, image generation was executed on Google Colab using a T4 GPU.

The Colab notebook:

- Clones this repository.
- Installs dependencies.
- Loads JSON analyses from `data/processed`.
- Builds prompts using the logic described above.
- Runs both `sd-turbo` and `sdxl-turbo`.
- Saves all images and `image_generation_log.csv` back into the repo structure.

**Colab link:**  
`https://colab.research.google.com/drive/1Q_K06jRqUam4Gcgz8l44NdwZ0FCMd8r0`
## 6. Q4 – Agentic AI Workflow

Q4 connects Q1–Q3 into a simple agentic pipeline that runs end-to-end with one function call instead of manual notebook execution.

### 6.1. Agent Roles

1. **ReviewAnalysisAgent**  
   Converts raw reviews into structured JSON using `gpt-4o-mini`.  
   *Inputs:* product metadata + review CSVs  
   *Outputs:* `productX_analysis.json`

2. **PromptBuilderAgent**  
   Transforms JSON analysis into three prompts per product (clean, detail, in_use).  
   *Inputs:* product JSON + metadata  
   *Outputs:* dictionary of prompts per product

3. **DiffusionImageAgent**  
   Generates images via `stabilityai/sd-turbo` and `stabilityai/sdxl-turbo`.  
   *Outputs:* 18 images + `image_generation_log.csv`

4. **EvaluationAgent**  
   Loads the log into a DataFrame for inspection or comparison.

### 6.2. Workflow Execution

```text
ReviewAnalysisAgent
        ↓
PromptBuilderAgent
        ↓
DiffusionImageAgent
        ↓
EvaluationAgent
```
Q4 is implemented as a set of Python functions that mirror these agent roles and can be orchestrated by a single driver (for example, a `run_pipeline()` function in a notebook or script).

- Automation: one call performs review analysis → prompt creation → image generation → logging

- Modularity: each agent has a single responsibility and clean input/output

- Reproducibility: all artifacts (JSON, prompts, images, logs) are stored under data/

- Extendable: new models, validators, or comparison methods can plug into the same sequence
