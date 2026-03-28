# Industry 4.0 Maturity Assessment Agent

> An AI-powered CLI agent that evaluates a manufacturing company's Industry 4.0 maturity across 7 key dimensions — grounded in the **IMPULS Industrie 4.0 Readiness Model** — and delivers scored, context-aware recommendations via a RAG pipeline backed by GPT-4o-mini.

## Table of Contents

- [Overview](#overview)
- [Background & Research](#background--research)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Assessment Dimensions](#assessment-dimensions)
- [Scoring Model](#scoring-model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Example Output](#example-output)
- [Limitations & Future Work](#limitations--future-work)


## Overview

The **Industry 4.0 Maturity Assessment Agent** is an interactive command-line tool that guides manufacturing companies through a structured, 105-question assessment. For every answer given, the agent retrieves relevant context from the IMPULS documentation using vector similarity search, then uses GPT-4o-mini to score the response (0–5) and produce a tailored recommendation.

The result is a per-dimension and overall maturity score, classifying the company as a **Newcomer**, **Learner**, or **Leader** — directly mirroring the IMPULS framework used by the German mechanical engineering industry (VDMA).


## Background & Research

This project is built on top of the **IMPULS Industrie 4.0 Readiness Study** (2015), commissioned by VDMA's IMPULS-Stiftung and conducted by IW Consult and FIR at RWTH Aachen University. The study defines a 6-level readiness model across 6 core dimensions (expanded here to 7 with CSR).

Key concepts driving this project:

- The IMPULS Readiness Model classifies companies as **Outsiders (0)**, **Beginners (1)**, **Intermediate (2)**, **Experienced (3)**, **Expert (4)**, or **Top Performers (5)**
- The model covers dimensions from smart factory hardware to data-driven services and employee skills
- Less than 6% of surveyed German mechanical engineering companies qualified as "Leaders" at the time of the study — highlighting the need for accessible, actionable self-assessment tools

This agent automates and extends that self-assessment into a conversational, AI-augmented experience.


## Features

- **RAG-powered analysis** — chunks and embeds the IMPULS PDF into a ChromaDB vector store; retrieves the most relevant passages per question before scoring
- **Context-aware scoring** — GPT-4o-mini scores each answer in light of the company's size, industry type, manufacturing type, and location
- **105 quantitative questions** — responses are validated as either percentages (0–100) or numeric counts/durations, avoiding ambiguous free-text
- **7 assessment dimensions** with multiple subdimensions each
- **Per-dimension and overall scoring** — all results persisted to SQLite and summarised at the end of the session
- **Input validation** — malformed or out-of-range answers are rejected and the user is re-prompted
- **Clean, structured output** — each response returns a Score, Justification, and Recommendation


## Architecture

┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                        │
│            (company info → dimensions → questions)          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               Industry40AssessmentAgent                     │
│                                                             │
│  ┌──────────────┐    ┌───────────────┐   ┌──────────────┐  │
│  │  SQLite DB   │    │  ChromaDB     │   │  OpenAI API  │  │
│  │              │    │  Vector Store │   │  GPT-4o-mini │  │
│  │ • sessions   │    │               │   │              │  │
│  │ • questions  │◄──►│ IMPULS PDF    │◄──► Scoring +    │  │
│  │ • responses  │    │ embeddings    │   │ Recommen-    │  │
│  │ • scores     │    │ (RAG)         │   │ dations      │  │
│  └──────────────┘    └───────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────────┘

**Flow per question:**
1. User submits a numeric answer
2. Agent validates and cleans the input
3. ChromaDB similarity search retrieves the 2 most relevant IMPULS passages for the dimension/subdimension
4. A structured prompt (company context + question + IMPULS context) is sent to GPT-4o-mini
5. The model returns `Score / Justification / Recommendation`
6. The result is parsed, stored in SQLite, and printed to the user


## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | ChromaDB (LangChain wrapper) |
| PDF Parsing | PyPDF2 |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Database | SQLite 3 |
| Config | python-dotenv |


## Assessment Dimensions

The agent covers **7 dimensions** with **105 questions** total (15 per dimension, 3 per subdimension):

| # | Dimension | Subdimensions |
|---|---|---|
| 1 | **Smart Factory** | Equipment Infrastructure, Data Collection & Usage, System Integration, Digital Twin & Simulation, Cybersecurity |
| 2 | **Smart Operations** | Vertical & Horizontal Integration, Supply Chain Visibility, Autonomous Processes, Predictive Analytics, Real-time Process Optimization |
| 3 | **Smart Products** | Product Digitalization, Product Integration, Data Analytics, Product Customization, Product Security |
| 4 | **Data-driven Services** | Service Portfolio, Data Collection & Analysis, Predictive Services, *(+ more)* |
| 5 | **Employees** | Digital Skills, Training & Development, *(+ more)* |
| 6 | **Strategy & Organization** | Digital Strategy, Organization Structure, Innovation Management, Digital Governance, Investment & Resources |
| 7 | **Corporate Social Responsibility** | Environmental Impact, Social Impact, Sustainable Operations, Ethical Technology, Responsible Innovation |


## Scoring Model

Each question is scored **0–5** by the LLM, calibrated to the IMPULS readiness levels:

| Score | IMPULS Level | Company Type |
|---|---|---|
| 0 | Outsider | No requirements met |
| 1 | Beginner | Pilot-level initiatives only |
| 2–2.5 | Intermediate / Learner | First steps in implementation |
| 3 | Experienced | Strategy defined, investments underway |
| 4 | Expert | Advanced implementation |
| 5 | Top Performer | Full Industry 4.0 vision realised |

The **dimension score** is the average of all question scores within that dimension. The **overall score** is the average across all 105 responses. Both are stored in SQLite at the end of the session.


## Project Structure

industry40-assessment-agent/
│
├── intelligent_agent.py          # Core agent class + CLI runner
├── assessment_questions.json     # All 105 questions across 7 dimensions
├── impuls_documentation.pdf      # IMPULS Readiness Study (knowledge base)
├── industry40_assessment.db      # SQLite database (auto-created)
├── impuls_documentation/         # ChromaDB vector store (auto-created)
├── .env                          # API keys (not committed)
├── requirements.txt              # Python dependencies
└── README.md
```


## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/industry40-assessment-agent.git
cd industry40-assessment-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Then edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### Requirements

Create a `requirements.txt` with the following:

```
openai
langchain
langchain-community
chromadb
PyPDF2
python-dotenv
```

---

## Usage

```bash
python intelligent_agent.py
```

The agent will:

1. **Load the knowledge base** — parse the IMPULS PDF and build the ChromaDB vector store (first run only; cached after that)
2. **Collect company profile** — prompt for company name, size, industry type, manufacturing type, and location
3. **Run the assessment** — walk through all 7 dimensions and their questions sequentially
4. **Print live feedback** — score, justification, and recommendation after each answer
5. **Show the final summary** — per-dimension scores and overall maturity score

**Example session start:**

```
Initializing assessment agent...
Knowledge base initialized successfully!

Welcome to Industry 4.0 Maturity Assessment
Company name: Acme Manufacturing GmbH

Company Information:
-------------------
Available company sizes:
  Small:  20–99 employees,   Revenue: < €10M
  Medium: 100–499 employees, Revenue: €10M–€50M
  Large:  500+ employees,    Revenue: > €50M

Enter company size: medium
...
```

**Example question interaction:**

```
=== Smart Factory ===

-- Equipment Infrastructure --

What percentage of your manufacturing equipment is capable of
machine-to-machine (M2M) communication?
(Answer in %)
Your answer: 35

Score: 2.5/5
Analysis: A 35% M2M capability places the company at an Intermediate
level per IMPULS criteria. While foundational connectivity exists,
more than half of equipment remains isolated.
Recommendation: Prioritise upgrading legacy CNC and assembly line
equipment with OPC-UA compatible interfaces. A phased retrofit
programme targeting the highest-throughput machines first will
yield the fastest readiness gains for a medium-sized manufacturer.
```

---

## Database Schema

All assessment data is persisted in `industry40_assessment.db`:

| Table | Key Columns |
|---|---|
| `assessment_sessions` | `session_id`, `company_name`, `company_size`, `industry_type`, `manufacturing_type`, `location`, `created_at` |
| `dimensions` | `dimension_id`, `name` |
| `subdimensions` | `subdimension_id`, `dimension_id`, `name` |
| `questions` | `question_id`, `subdimension_id`, `dimension_id`, `question_text`, `response_type`, `unit` |
| `question_responses` | `session_id`, `question_id`, `response_value`, `question_score`, `feedback`, `ai_analysis` |
| `company_sizes` | `size_id`, `employee_range`, `revenue_range` |
| `industry_types` | `name` |
| `manufacturing_types` | `name` |

---

## Example Output

```
=== Assessment Summary ===
Company: Acme Manufacturing GmbH
Size: medium
Industry: Mechanical Engineering

Dimension Scores:
  Smart Factory           : 2.8 / 5
  Smart Operations        : 2.3 / 5
  Smart Products          : 1.9 / 5
  Data-driven Services    : 1.2 / 5
  Employees               : 3.1 / 5
  Strategy & Organization : 2.6 / 5
  CSR                     : 2.0 / 5

Overall Industry 4.0 Maturity Score: 2.27 / 5
Classification: Learner

Assessment completed!
```

---

## Limitations & Future Work

**Current limitations:**
- CLI only — no web interface
- The LLM prompt instructs the model not to invent answers when input is unclear, but edge cases may still occur
- The vector store is rebuilt from scratch if the persist directory is deleted
- No authentication or multi-user session management

**Planned improvements:**
- [ ] Web dashboard with radar/spider chart visualisation of dimension scores
- [ ] Benchmarking against anonymised industry averages (as in the original IMPULS study)
- [ ] Export assessment report to PDF
- [ ] Support for follow-up questions and clarification within a session
- [ ] Fine-tuned scoring model trained on validated IMPULS assessment data

---

## Acknowledgements

This project is grounded in the **IMPULS Industrie 4.0 Readiness** study (Lichtblau et al., 2015), commissioned by VDMA's IMPULS-Stiftung and conducted by IW Consult (Cologne Institute for Economic Research) and FIR at RWTH Aachen University. All domain knowledge, readiness levels, and dimension definitions are derived from that research.

---

## License

This project is for educational and portfolio purposes.

---

*Built with domain research, structured thinking, and AI-assisted implementation.*
