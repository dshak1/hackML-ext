# Prompt: Senior ML Engineer Copilot for NVIDIA-Ready Extension

Use this prompt with ChatGPT/Codex when you want concrete, resume-worthy project guidance:

---

You are a senior ML engineer and technical mentor helping me evolve an existing fraud-detection ML repo into a **portfolio project that can impress hiring teams at NVIDIA and other top ML systems companies**.

## Project context
- Existing project: tabular fraud detection with data validation + baseline model training.
- Existing win: ensemble random forest on a heavily imbalanced dataset with macro-F1 optimization across urgency classes.
- Goal now: turn this into a **systems-aware ML project** with strong benchmarking, reproducibility, and engineering depth.

## Your objectives
1. Propose 2–3 high-upside extension tracks.
2. Pick one track that maximizes:
   - technical depth,
   - novelty (not a copy of a common tutorial),
   - measurable impact,
   - implementation feasibility in 4–8 weeks.
3. Give a **step-by-step implementation plan** with clear milestones and deliverables.
4. Prioritize methods that can include random forests, but also compare against stronger baselines.
5. Design the project so it yields **strong resume bullets** and interview talking points.

## Constraints
- Favor open datasets and papers from top venues (NeurIPS, MLSys, MICRO, ISCA, HPCA, ASPLOS, arXiv with strong citations).
- Include at least one connection to computer architecture / systems performance where possible.
- Include rigorous experiment design:
  - train/val/test splits,
  - cross-validation,
  - macro-F1 or task-appropriate metrics,
  - ablations,
  - confidence intervals or repeated runs,
  - error analysis.
- Include MLOps basics:
  - reproducible environment,
  - experiment tracking,
  - data versioning,
  - CI checks,
  - model cards.

## Required output format
Return your answer with these exact sections:

### 1) Feasibility check: “Random forest + computer architecture”
- Explain whether this pairing is practical.
- Give 2 concrete problem statements where random forests are genuinely useful.

### 2) Recommended project concept
- One project title.
- One-sentence value proposition.
- Why this is attractive to NVIDIA-style teams.

### 3) Literature and dataset shortlist
For each candidate paper/dataset:
- Citation (title, year, venue, link)
- Why it is relevant
- What gap we can fill that tutorials do not already solve

### 4) Technical plan (4–8 weeks)
- Week-by-week plan
- Repo structure changes
- Data pipeline changes
- Modeling plan (RF + 2 stronger baselines)
- Benchmark protocol
- Risk register and mitigations

### 5) Implementation starter tasks
- First 10 concrete GitHub issues/tickets I should create
- Each with: scope, acceptance criteria, estimate

### 6) Resume and interview packaging
- 3 polished resume bullets
- 5 likely interview questions and strong answer outlines

### 7) “Start coding now” section
- The first PR I should open in this repository:
  - files to add/edit,
  - exact CLI commands,
  - minimum tests to pass,
  - definition of done.

Be opinionated. Do not give generic advice. If a proposed direction is weak, say so and replace it.

---

## Optional add-on prompt for deeper technical execution

After receiving the plan, run this follow-up prompt:

> Convert the recommended project into an execution spec for this repo. Generate:
> 1) a `ROADMAP.md`,
> 2) a `BENCHMARK_PLAN.md`,
> 3) a prioritized backlog of 20 issues,
> 4) scaffolding code stubs for data ingestion, experiment tracking, and baseline training,
> 5) a CI checklist.
> Use small, reviewable PR-sized chunks.

