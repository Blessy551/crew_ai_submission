# CrewAI Agents API

One FastAPI server that exposes 10 CrewAI projects as `POST /agent1` … `POST /agent10`. All routes share the same JSON contract.

**Live URL:** https://crew-ai-submission.onrender.com (Swagger UI with a sample input per route: https://crew-ai-submission.onrender.com/docs)
(Free hosting tiers may sleep; the first request can take ~30 s to wake the server.)

## API contract

Every route: `POST /agentN`, `Content-Type: application/json`

Request:

```json
{ "input": "..." }
```

Response (200):

```json
{ "output": "..." }
```

Errors: `400` with `{"detail": "..."}` for empty or invalid input, `500` with `{"detail": "..."}` if the run fails.
Interactive docs (Swagger) are available at `/docs`.

## Routes

| Route | Project | `input` should contain | Built with |
|---|---|---|---|
| `/agent1` | Blog post writer | A topic, e.g. `Benefits of urban rooftop gardens` | Crew: Researcher (DuckDuckGo search), Writer, Editor (FileWriterTool) |
| `/agent2` | Competitor analysis | One or more `http(s)` URLs | Hierarchical Crew with a manager LLM; ScrapeWebsiteTool, search, FileWriterTool |
| `/agent3` | Meeting notes | A meeting transcript | Crew: FileReadTool, 3 parallel (async) analysts, Formatter (FileWriterTool) |
| `/agent4` | Data report | CSV text with a header row and at least one data row | Crew: custom CSV stats tool, structured (Pydantic) output, Analyst / Insight Writer / Reviewer |
| `/agent5` | Customer support | A customer message | Flow: categorize → router → billing / access / other → resolve |
| `/agent6` | Resume screening | `RESUME: ... JOB DESCRIPTION: ...` (job description optional) | Flow: score 0–10 → router (≥7 shortlist, else rejection draft) → results saved to CSV |
| `/agent7` | Bug / feature / question triage | A user report | Flow: classify → router → bug / feature / answer crew (ScrapeWebsiteTool + search) |
| `/agent8` | Content repurposing | Article text, or a bare URL | Flow: extract key points → LinkedIn, Twitter and newsletter crews in parallel → `and_` publish pack |
| `/agent9` | Lead qualification | `Company https://company-site.com` | Flow: capture → scrape/enrich → router hot / warm / cold → outreach / nurture / archive → log to CSV |
| `/agent10` | Approval report | A report topic | Flow: research crew → writing crew → automated review → router approve / revise → publish |

Notes:

- `/agent10` uses an **automated reviewer** in place of the human approval step. The output starts with `Status: approved (reviewed automatically)` or `Status: published after revision (reviewed automatically)`.
- Routes that write files (`output/*.md`, `output/*.csv`) also return their content in `output`, because hosted disks are ephemeral.

## Example

```bash
curl -X POST http://127.0.0.1:8000/agent1 \
  -H "Content-Type: application/json" \
  -d '{"input": "Benefits of urban rooftop gardens"}'
```

```json
{ "output": "# Benefits of Urban Rooftop Gardens ..." }
```

Sample inputs used for testing:

| Route | Sample `input` |
|---|---|
| agent1 | `Benefits of urban rooftop gardens` |
| agent2 | `https://www.notion.com https://evernote.com https://obsidian.md` |
| agent3 | `Priya: Let's review the Q3 launch. Rahul: Backend is on track but QA needs two more days. Priya: Okay, we decided to move the launch from Oct 10 to Oct 14. Rahul: I will finish the API tests by Oct 8.` |
| agent4 | `date,region,product,units,revenue\n2026-01-05,North,Widget,10,100.0\n2026-01-06,South,Widget,5,50.0\n2026-01-07,North,Gadget,3,90.0` |
| agent5 | `I was charged twice for my subscription this month, please refund one payment.` |
| agent6 | `RESUME: Jane Doe. 5 years Python, FastAPI, AWS, led a team of 4. JOB DESCRIPTION: Senior Python backend engineer, FastAPI, cloud experience.` |
| agent7 | `App crashes with a NullPointerException every time I tap Save on the profile page after uploading a photo larger than 5MB. Android 14.` |
| agent8 | Any article text, or `https://example.com/some-article` |
| agent9 | `Stripe https://stripe.com` |
| agent10 | `Benefits of a four-day work week` |

## How it works

- **Framework:** FastAPI served by **uvicorn**. All routing lives in `app.py`.
- **Agents:** CrewAI. Routes 1–4 are Crews, routes 5–10 are CrewAI Flows (`@start`, `@listen`, `@router`, `or_`, `and_`).
- **Tools:** `FileWriterTool`, `FileReadTool`, `ScrapeWebsiteTool` (crewai-tools), plus two custom tools: a DuckDuckGo search (`ddgs`) and a CSV stats tool (stdlib `csv`).
- **LLM:** Gemini, default `gemini/gemini-3.1-flash-lite-preview`. On 503 / 429 errors a request is retried on fallback Gemini models. Override the primary model with the `GEMINI_MODEL` env var.

## Run locally

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file next to `app.py`:

```
GEMINI_API_KEY=your-key-here
```

Start the server (the first start takes ~30 s because CrewAI is slow to import):

```bash
python -m uvicorn app:app --port 8000
```

## Deploy (Render / Railway)

- Runtime: Python 3.11
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Environment variable: `GEMINI_API_KEY`
- Pick a plan and host without a short request timeout (see limitations).

## Testing in Postman

1. Method **POST**, URL `<base>/agentN`.
2. Body → raw → JSON: `{"input": "..."}`.
3. Settings → request timeout: set to 0 (no limit); some routes take minutes.
4. Pass = HTTP 200 and a non-empty `output`.

Approximate local run times: agent1 ~1 min, agent2 ~7 min, agent3 ~1 min, agent4 ~1 min, agent5 ~20 s, agent6 ~2 min, agent7 ~1 min, agent8 ~30 s, agent9 ~2 min, agent10 ~2 min.

## Known limitations

- **Gemini quota:** the free tier allows roughly 20 requests per day per model. Enable billing on the key for sustained use.
- **Preview models** sometimes return 503; the server retries on fallback models, which adds delay.
- **`/agent2` is slow** (about 7 minutes) and can exceed a host's or client's request timeout.
- **Outputs vary** between runs (LLM output is non-deterministic), e.g. the resume score can differ by a point.
- **Files in `output/`** are not persistent on hosted platforms; use the `output` field of the response.
- **No authentication or rate limiting.** Each call spends the owner's Gemini quota, so don't publish the URL widely.
