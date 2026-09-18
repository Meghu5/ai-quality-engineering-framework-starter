# AI Quality Engineering Framework

Portfolio project for testing LLM, RAG and agentic applications using Python, Pytest, Playwright, REST API automation and GitHub Actions.

## Quality layers
- Deterministic API/UI validation
- LLM semantic evaluation: relevance, accuracy, completeness, hallucination
- RAG retrieval and groundedness checks
- Agent tool-selection, arguments, completion and recovery
- Defensive AI-security tests: prompt injection, jailbreaks and sensitive-data leakage
- CI quality gates and HTML/JSON evidence

## Status
**Portfolio Project - In Progress**

## Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
pytest -v
```

Set `CHAT_BASE_URL` to the application under test. The first milestone can use the included mock/demo service.

## Git workflow
```powershell
git init
git branch -M main
git add .
git commit -m "chore: initialize AI quality engineering framework"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Use small conventional commits such as:
- `feat: add deterministic chat API tests`
- `feat: add Playwright chatbot coverage`
- `feat: add semantic evaluator`
- `feat: add RAG grounding tests`
- `feat: add agent tool validation`
- `test: add prompt injection dataset`
- `ci: add GitHub Actions quality gate`
- `docs: update architecture and demo`

## Project roadmap
1. Base API client + fixtures + deterministic gates
2. Playwright UI flow
3. Dataset-driven LLM evaluator
4. RAG retrieval/grounding/faithfulness
5. Agent/tool-call validation
6. Defensive AI security
7. CI/CD, reports and quality trends
8. README/demo/resume positioning

Do not mark the project complete until the repository runs reproducibly and CI passes.
