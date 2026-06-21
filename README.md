# POD-JD Intelligence

> **Automated placement email → multi-agent JD analysis → student prep brief + faculty curriculum recommendations**
>
> Built for MITAOE — CSE (AI & ML) Department  
> Stack: **LangGraph · Groq (Llama 3.3 70B) · LangSmith · FastMCP · IMAP/SMTP**

---

## What it does

1. **Polls** the POD (Training & Placement Office) mailbox via IMAP
2. **Filters** — only processes emails from the configured allow-list of POD senders
3. **Extracts** the Job Description from the email body and any attached PDF
4. **Parses** it into a structured JSON (company, role, skills, responsibilities, CTC, deadline …)
5. Runs **two parallel agents**:
   - **Student-Prep Agent** — 2-week prep plan, skill gaps, 10 interview questions, projects to build
   - **Faculty-Curriculum Agent** — CO-PO-aware syllabus gaps, lab approaches, project ideas, quick-win actions
6. **Saves drafts** locally (`.json`); dispatches via SMTP only after human approval
7. **Traces every LLM call** to LangSmith for monitoring and debugging

---

## Architecture

```
POD Email (IMAP)
      │
      ▼  validate sender + subject keyword
 ┌────────────┐
 │  Ingest    │  extract PDF text (pdfplumber → pypdf fallback)
 └─────┬──────┘
       │
       ▼
 ┌────────────┐
 │  JD Parser │  Groq Llama 3.3-70B → structured JSON
 └─────┬──────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
Student  Faculty           ← both run independently (LangGraph fan-out)
 Prep    Curriculum
 Agent    Agent
   │       │
   └───┬───┘
       ▼
 ┌────────────┐
 │   Notify   │  save drafts → optionally send via SMTP
 └────────────┘
       │
  output/jds/      output/drafts/
  {company}.json   student_{id}.json   faculty_{id}.json
```

All LLM calls are traced in **LangSmith** → [smith.langchain.com](https://smith.langchain.com)

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Groq API key | [console.groq.com](https://console.groq.com) (free tier) |
| LangSmith API key | [smith.langchain.com](https://smith.langchain.com) (free tier) |
| Email credentials | IMAP App Password (Office 365 or Gmail) |
| Docker *(optional)* | 24+ |

---

## Quick Start

### Local

```bash
git clone https://github.com/mitaoe-aiml/pod-jd-intelligence.git
cd pod-jd-intelligence

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# ↑ edit .env — add GROQ_API_KEY, LANGCHAIN_API_KEY, IMAP creds
```

**Run once:**
```bash
python main.py run
```

**Run and send immediately** (skip dry-run — use only after testing):
```bash
python main.py run --send
```

### Docker

```bash
cp .env.example .env          # fill in credentials
docker compose up -d          # starts MCP server + daily scheduler (9 AM, 3 PM)
docker compose run --rm pipeline   # manual one-shot run
docker compose logs -f        # live logs
```

---

## Configuration (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key — get at console.groq.com |
| `LLM_MODEL` | | Groq model ID (default: `llama-3.3-70b-versatile`) |
| `LANGCHAIN_TRACING_V2` | | `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | | LangSmith API key |
| `LANGCHAIN_PROJECT` | | LangSmith project name (default: `pod-jd-intelligence`) |
| `IMAP_HOST` | ✅ | IMAP server (e.g. `outlook.office365.com`) |
| `IMAP_USER` | ✅ | Email address to poll |
| `IMAP_PASSWORD` | ✅ | App password (NOT login password) |
| `SMTP_HOST` | ✅ | SMTP server for sending approved drafts |
| `POD_ALLOWED_SENDERS` | ✅ | Comma-separated list of trusted POD email addresses |
| `STUDENT_LIST_EMAIL` | ✅ | Recipient address for student briefs |
| `FACULTY_LIST_EMAIL` | ✅ | Recipient address for faculty recommendations |
| `DRY_RUN` | | `true` (default) = save drafts, don't send |
| `JD_SUBJECT_KEYWORDS` | | Comma-separated subject filters |

### Getting an App Password

**Office 365 / Outlook:**  
Settings → Security → Advanced security options → App passwords

**Gmail:**  
Google Account → Security → 2-Step Verification → App passwords

---

## CLI Reference

```bash
python main.py run              # process new POD emails (dry_run from .env)
python main.py run --send       # process + send drafts immediately
python main.py list-jds         # list all parsed JDs
python main.py list-drafts      # list pending draft emails
python main.py approve <path>   # send a specific draft via SMTP
python main.py status           # pipeline stats
```

---

## MCP Server (Claude Code integration)

Run the FastMCP server:
```bash
python mcp_server.py
```

Add to `~/.claude/mcp_servers.json`:
```json
{
  "mcpServers": {
    "pod-jd-intelligence": {
      "command": "python",
      "args": ["/path/to/pod_jd_intelligence/mcp_server.py"],
      "env": { "PYTHONPATH": "/path/to/pod_jd_intelligence" }
    }
  }
}
```

Available tools from Claude Code:
| Tool | Description |
|---|---|
| `run_pipeline(dry_run=True)` | Trigger full pipeline |
| `list_processed_jds()` | View all parsed JDs |
| `list_pending_drafts()` | View drafts awaiting approval |
| `approve_draft(path)` | Send an approved draft |
| `get_pipeline_status()` | Stats + config info |

---

## Service Deployment

### Linux (systemd)

```bash
# Create user
sudo useradd -r -s /sbin/nologin pod-jd

# Deploy
sudo mkdir -p /opt/pod-jd-intelligence
sudo cp -r . /opt/pod-jd-intelligence/
sudo chown -R pod-jd:pod-jd /opt/pod-jd-intelligence

# Create venv as that user
sudo -u pod-jd python3 -m venv /opt/pod-jd-intelligence/venv
sudo -u pod-jd /opt/pod-jd-intelligence/venv/bin/pip install -r /opt/pod-jd-intelligence/requirements.txt

# Install service + timer
sudo cp services/pod-jd.service /etc/systemd/system/
sudo cp services/pod-jd.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pod-jd.timer

# Check
systemctl list-timers pod-jd.timer
journalctl -u pod-jd.service -f
```

### macOS (launchd)

```bash
cp services/com.mitaoe.pod-jd.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.mitaoe.pod-jd.plist

# Logs
tail -f /tmp/pod-jd.out.log
tail -f /tmp/pod-jd.err.log
```

---

## LangSmith Monitoring

Once `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set, every pipeline run
creates a trace in your LangSmith project. You can:

- See end-to-end latency per node (ingest → parse → agents → notify)
- Inspect every Groq prompt and response
- Compare runs across different JDs
- Set up alerts for errors or high latency

**Dashboard:** [smith.langchain.com](https://smith.langchain.com) → select project `pod-jd-intelligence`

---

## Project Structure

```
pod_jd_intelligence/
├── .env.example             # template — copy to .env and fill in
├── .gitignore
├── .dockerignore
├── requirements.txt
├── config.py                # central config, reads .env
├── main.py                  # CLI entry point
├── mcp_server.py            # FastMCP server (5 tools)
├── Dockerfile               # multi-stage build
├── docker-compose.yml       # MCP server + scheduler + one-shot pipeline
├── services/
│   ├── pod-jd.service       # systemd service
│   ├── pod-jd.timer         # systemd timer (9 AM + 3 PM)
│   └── com.mitaoe.pod-jd.plist  # macOS launchd agent
├── tools/
│   ├── llm.py               # Groq client + LangSmith @traceable wrapper
│   ├── email_reader.py      # IMAP polling + POD sender validation
│   └── pdf_tool.py          # pdfplumber / pypdf text extraction
├── workflow/
│   ├── state.py             # PipelineState TypedDict + JDStructured schema
│   ├── graph.py             # LangGraph StateGraph
│   └── nodes/
│       ├── ingest.py        # PDF extraction node
│       ├── parse_jd.py      # JD Parser agent (Groq, JSON mode)
│       ├── student_prep.py  # Student-Prep agent (Groq)
│       ├── faculty_reco.py  # Faculty-Curriculum agent (Groq)
│       └── notify.py        # Draft composer + SMTP sender
└── output/
    ├── jds/                 # parsed JD JSONs
    └── drafts/              # draft email JSONs (pending approval)
```

---

## Groq Models

| Model | Use case | Speed |
|---|---|---|
| `llama-3.3-70b-versatile` | Default — best quality for JD parsing + curriculum reco | Fast |
| `llama-3.1-8b-instant` | Cost/speed optimised — for high-volume testing | Very fast |
| `mixtral-8x7b-32768` | Long context (32K) — for very long JD PDFs | Fast |

Change model per-run: set `LLM_MODEL=mixtral-8x7b-32768` in `.env`.

---

## Security Notes

- `.env` is git-ignored — never commit credentials
- `DRY_RUN=true` is the default — emails are never sent without explicit approval
- Only emails from `POD_ALLOWED_SENDERS` are processed — all other senders are silently ignored
- The systemd service runs as a dedicated non-root user (`pod-jd`)
- Docker image runs as non-root user (`podjd`)

---

## License

MIT — MITAOE CSE-AIML Department
