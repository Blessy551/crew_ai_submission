import csv
import io
import os
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime

from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, and_, listen, or_, router, start
from crewai.tools import tool
from crewai_tools import FileReadTool, FileWriterTool, ScrapeWebsiteTool
from ddgs import DDGS
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="CrewAI Projects")


class In(BaseModel):
    input: str


class Out(BaseModel):
    output: str
    model_config = {"json_schema_extra": {"example": {"output": "# Final answer..."}}}


def ex(text: str):
    """Per-route sample request body shown in /docs and /openapi.json."""
    return Body(openapi_examples={"sample": {"summary": "Sample input", "value": {"input": text}}})


# Free tier = ~20 requests/day PER MODEL and preview models 503 often, so retries walk this list.
MODELS = [
    os.getenv("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite-preview"),
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-flash-lite-latest",
]
model_idx: ContextVar[int] = ContextVar("model_idx", default=0)


def llm() -> LLM:
    return LLM(model=MODELS[model_idx.get()], api_key=os.environ["GEMINI_API_KEY"])


@tool("DuckDuckGoSearch")
def search(query: str) -> str:
    """Search the web with DuckDuckGo and return titles, links and snippets."""
    results = DDGS().text(query, max_results=5)
    return "\n".join(f"{r['title']} - {r['href']}\n{r['body']}" for r in results)


def run(fn, text: str) -> Out:
    if not text.strip():
        raise HTTPException(400, "input must not be empty")
    for attempt in range(len(MODELS)):
        try:
            return Out(output=str(fn(text)))
        except HTTPException:
            raise
        except Exception as e:
            busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE"))
            if busy and attempt < len(MODELS) - 1:
                model_idx.set(attempt + 1)
                time.sleep(5)
                continue
            raise HTTPException(500, f"{type(e).__name__}: {e}")


# ---------- agent1: Blog Post Crew ----------
def blog_post(topic: str) -> str:
    m = llm()
    researcher = Agent(
        role="Researcher", goal="Find accurate, current facts about the topic",
        backstory="A careful web researcher who cites what they find.",
        tools=[search], llm=m,
    )
    writer = Agent(
        role="Writer", goal="Write an engaging blog post from research notes",
        backstory="A clear, friendly blogger.", llm=m,
    )
    editor = Agent(
        role="Editor", goal="Polish tone, fix errors, keep facts consistent with the research",
        backstory="A strict but fair copy editor.", tools=[FileWriterTool()], llm=m,
    )
    t1 = Task(
        description=f"Research the topic: {topic}. Use the search tool. Collect 5-8 key facts.",
        expected_output="Bullet list of facts with source links.", agent=researcher,
    )
    t2 = Task(
        description="Write a ~600 word blog post in markdown using the research.",
        expected_output="A markdown blog post with a title and sections.", agent=writer,
    )
    t3 = Task(
        description=(
            "Edit the draft for tone, grammar and factual consistency. Then save the final "
            "post with the file writer tool: filename 'blog_post.md', directory 'output', "
            "overwrite true. Finally reply with the full final post."
        ),
        expected_output="The full final blog post in markdown (also saved to output/blog_post.md).",
        agent=editor,
    )
    crew = Crew(agents=[researcher, writer, editor], tasks=[t1, t2, t3],
                process=Process.sequential)
    return crew.kickoff().raw


@app.post("/agent1", response_model=Out)
def agent1(body: In = ex("Benefits of urban rooftop gardens")):
    return run(blog_post, body.input)


# ---------- agent2: Competitor Analysis Crew (hierarchical) ----------
def competitor_analysis(text: str) -> str:
    urls = re.findall(r"https?://\S+", text)
    if not urls:
        raise HTTPException(400, "input must contain at least one http(s) URL")
    m = llm()
    scraper = Agent(
        role="Site Scraper", goal="Extract what each competitor sells and how they position it",
        backstory="Reads competitor websites and reports only what is on the page.",
        tools=[ScrapeWebsiteTool()], llm=m,
    )
    pricing = Agent(
        role="Pricing Analyst", goal="Summarise plans and prices for each competitor",
        backstory="Compares pricing tiers precisely and notes when prices are not public.", llm=m,
    )
    news = Agent(
        role="News Analyst", goal="Find recent news and launches for each competitor",
        backstory="Tracks competitor announcements on the web.", tools=[search], llm=m,
    )
    strategist = Agent(
        role="Strategist", goal="Turn the findings into a clear SWOT-style competitor report",
        backstory="A product strategist who writes concise, decision-ready reports.",
        tools=[FileWriterTool()], llm=m,
    )
    t1 = Task(
        description=f"Scrape these competitor sites and summarise products and positioning:\n{text}",
        expected_output="Per competitor: products, target customers, key claims.",
    )
    t2 = Task(
        description=f"Using the scraped info, list plans and pricing per competitor. Competitors:\n{text}",
        expected_output="Per competitor: plans, prices, or 'not public'.", context=[t1],
    )
    t3 = Task(
        description=f"Search for recent news about each competitor:\n{text}",
        expected_output="Per competitor: 2-3 recent items with dates and links.",
    )
    t4 = Task(
        description=(
            "Write a markdown competitor report with a comparison table and a SWOT table per "
            "competitor. Save it with the file writer tool: filename 'report.md', directory "
            "'output', overwrite true. Finally reply with the full report."
        ),
        expected_output="The full markdown report (also saved to output/report.md).",
        context=[t1, t2, t3],
    )
    crew = Crew(
        agents=[scraper, pricing, news, strategist], tasks=[t1, t2, t3, t4],
        process=Process.hierarchical, manager_llm=m,
    )
    return crew.kickoff().raw


@app.post("/agent2", response_model=Out)
def agent2(body: In = ex("https://www.notion.com https://evernote.com")):
    return run(competitor_analysis, body.input)


# ---------- agent3: Meeting Notes Crew (parallel tasks) ----------
def meeting_notes(transcript: str) -> str:
    os.makedirs("output", exist_ok=True)
    path = f"output/transcript_{uuid.uuid4().hex[:8]}.txt"  # FileReadTool needs a real file
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)
    m = llm()
    def analyst(role):  # one agent per parallel task: an agent can't run two tasks at once
        return Agent(
            role=role, goal="Extract exactly what was said, never invent details",
            backstory="Reads meeting transcripts closely and reports only what is in them.",
            tools=[FileReadTool()], llm=m,
        )

    formatter = Agent(
        role="Formatter", goal="Combine the extracted parts into clean meeting notes",
        backstory="Produces tidy, well-structured markdown notes.",
        tools=[FileWriterTool()], llm=m,
    )
    read = f"Read the transcript with the file read tool (file path: {path}). "
    parts = [
        Task(description=read + "Write a summary of the meeting as exactly 5 bullet lines, one sentence each.",
             expected_output="Exactly 5 bullet lines.", agent=analyst("Summarizer"), async_execution=True),
        Task(description=read + "List every action item with owner and due date ('unspecified' if missing).",
             expected_output="Bullet list: action - owner - due date.", agent=analyst("Action Finder"), async_execution=True),
        Task(description=read + "List every decision that was made.",
             expected_output="Bullet list of decisions.", agent=analyst("Decision Logger"), async_execution=True),
    ]
    final = Task(
        description=(
            "Combine the summary (keep it as exactly 5 bullet lines), action items and decisions "
            "into markdown notes with three headings. Save with the file writer tool: filename 'notes.md', directory 'output', "
            "overwrite true. Finally reply with the full notes."
        ),
        expected_output="The full markdown notes (also saved to output/notes.md).",
        agent=formatter, context=parts,
    )
    crew = Crew(agents=[t.agent for t in parts] + [formatter], tasks=[*parts, final], process=Process.sequential)
    try:
        return crew.kickoff().raw
    finally:
        os.remove(path)


@app.post("/agent3", response_model=Out)
def agent3(body: In = ex(
    "Priya: Let's review the Q3 launch. Rahul: Backend is on track but QA needs two more days. "
    "Priya: Okay, we decided to move the launch from Oct 10 to Oct 14. Rahul: I will finish the API tests by Oct 8. "
    "Sara: I'll update the landing page copy by Friday. Priya: Also we agreed to drop the dark mode feature from v1.")):
    return run(meeting_notes, body.input)


# ---------- agent4: Data Report Crew (structured output) ----------
class Metric(BaseModel):
    name: str
    value: float


class SalesStats(BaseModel):
    row_count: int
    columns: list[str]
    metrics: list[Metric]  # e.g. total sales, average order value, top segment total
    notes: str


@tool("CSVStats")
def csv_stats(path: str) -> str:
    """Compute statistics for an uploaded CSV file: shape, summary stats and group totals."""
    if not path.replace("\\", "/").startswith("output/data_"):
        return "Error: only files created for this request can be read."
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0]) if rows else []
    nums = [c for c in cols if all(_num(r[c]) is not None for r in rows)]
    out = [f"rows={len(rows)} columns={cols}"]
    for c in nums:
        v = [_num(r[c]) for r in rows]
        out.append(f"{c}: sum={sum(v):g} mean={sum(v) / len(v):g} min={min(v):g} max={max(v):g}")
    for c in (c for c in cols if c not in nums):
        groups = {}
        for r in rows:
            groups.setdefault(r[c], []).append(r)
        if 1 < len(groups) <= 20 and nums:  # totals per category
            out.append(f"Totals by {c}:")
            for k, rs in groups.items():
                sums = ", ".join(f"{n}={sum(_num(r[n]) for r in rs):g}" for n in nums)
                out.append(f"  {k} (rows={len(rs)}): {sums}")
    return "\n".join(out)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def data_report(csv_text: str) -> str:
    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(rows) >= 1 and len(rows[0]) >= 2
    except Exception:
        raise HTTPException(400, "input must be valid CSV text (header row + data rows)")
    os.makedirs("output", exist_ok=True)
    path = f"output/data_{uuid.uuid4().hex[:8]}.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_text)
    m = llm()
    analyst = Agent(
        role="Data Analyst", goal="Compute accurate statistics from the CSV",
        backstory="A careful analyst who only reports numbers the tools return.",
        tools=[FileReadTool(), csv_stats], llm=m,
    )
    writer = Agent(
        role="Insight Writer", goal="Explain the top 3 findings in plain language",
        backstory="Turns statistics into clear business insights.", llm=m,
    )
    reviewer = Agent(
        role="Reviewer", goal="Make sure every number in the report matches the statistics",
        backstory="A meticulous fact-checker.", tools=[csv_stats, FileWriterTool()], llm=m,
    )
    t1 = Task(
        description=f"Analyse the CSV file at {path} with the CSVStats tool (and FileReadTool if needed).",
        expected_output="Structured statistics.", agent=analyst, output_pydantic=SalesStats,
    )
    t2 = Task(
        description="Write the top 3 findings from the statistics, each with the supporting numbers.",
        expected_output="3 numbered findings.", agent=writer, context=[t1],
    )
    t3 = Task(
        description=(
            f"Check every number in the findings against the statistics (re-run CSVStats on {path} "
            "if unsure) and correct mistakes. Produce a markdown report with a short stats table "
            "and the 3 findings. Save it with the file writer tool: filename 'report.md', "
            "directory 'output', overwrite true. Finally reply with the full report."
        ),
        expected_output="The full markdown report (also saved to output/report.md).",
        agent=reviewer, context=[t1, t2],
    )
    crew = Crew(agents=[analyst, writer, reviewer], tasks=[t1, t2, t3], process=Process.sequential)
    try:
        return crew.kickoff().raw
    finally:
        os.remove(path)


@app.post("/agent4", response_model=Out)
def agent4(body: In = ex(
    "date,region,product,units,revenue\n2026-01-05,North,Widget,10,100.0\n2026-01-06,South,Widget,5,50.0\n"
    "2026-01-07,North,Gadget,3,90.0\n2026-01-08,East,Gadget,8,240.0\n2026-01-09,South,Gizmo,2,200.0\n"
    "2026-01-10,East,Widget,20,200.0\n2026-01-11,North,Gizmo,1,100.0")):
    return run(data_report, body.input)


# =====================================================================
# Flows (agent5-agent10)
# =====================================================================
class FlowState(BaseModel):
    result: str = ""  # final answer returned in {"output": ...}


def ask(prompt: str) -> str:
    """Direct LLM call, no Crew."""
    return str(llm().call(prompt)).strip()


def mini_crew(role: str, goal: str, task: str, expected: str, tools=None) -> str:
    """A one-agent, one-task Crew used inside Flow steps."""
    a = Agent(role=role, goal=goal, backstory=f"An expert {role}.", tools=tools or [], llm=llm())
    t = Task(description=task, expected_output=expected, agent=a)
    return Crew(agents=[a], tasks=[t]).kickoff().raw


def save(filename: str, content: str) -> None:
    FileWriterTool().run(filename=filename, content=content, directory="output", overwrite=True)


def log_row(filename: str, row: list) -> None:
    os.makedirs("output", exist_ok=True)
    with open(f"output/{filename}", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), *row])


def run_flow(flow_cls, **inputs) -> str:
    flow = flow_cls(suppress_flow_events=True)  # new instance per request = fresh state
    flow.kickoff(inputs=inputs)
    return flow.state.result


# ---------- agent5: Customer Support Flow (router + or_) ----------
class TicketState(FlowState):
    ticket: str = ""
    category: str = ""
    priority: str = "medium"


class SupportFlow(Flow[TicketState]):
    @start()
    def receive_ticket(self):
        self.state.ticket = self.state.ticket.strip()

    @listen(receive_ticket)
    def categorize_ticket(self):
        cats = ["billing", "account_access", "technical_issue", "feature_request", "other"]
        r = ask(
            f"Classify this support ticket. Reply with exactly one of: {', '.join(cats)}.\n\n"
            f"Ticket: {self.state.ticket}"
        ).lower()
        self.state.category = next((c for c in cats if c in r), "other")

    @router(categorize_ticket)
    def route_by_category(self):
        return self.state.category if self.state.category in ("billing", "account_access") else "other"

    @listen("billing")
    def handle_billing(self):
        self.state.priority = "high"

    @listen("account_access")
    def handle_access(self):
        self.state.priority = "high"

    @listen("other")
    def handle_other(self):
        self.state.priority = "medium"

    @listen(or_(handle_billing, handle_access, handle_other))
    def resolve_ticket(self):
        s = self.state
        reply = ask(
            f"You are a support agent. Ticket category: {s.category}. Priority: {s.priority}.\n"
            "Write a short, polite reply with concrete next steps. You cannot access any account "
            "systems, so never say you have escalated, forwarded, refunded or fixed anything; use "
            "future wording instead (e.g. 'this will be passed to our billing team') or say what "
            "the customer should do. End with the sign-off 'Customer Support Team' and no "
            f"[bracketed placeholders].\n\nTicket: {s.ticket}"
        )
        s.result = f"Category: {s.category}\nPriority: {s.priority}\n\n{reply}"


@app.post("/agent5", response_model=Out)
def agent5(body: In = ex("I was charged twice for my subscription this month, please refund one payment.")):
    return run(lambda t: run_flow(SupportFlow, ticket=t), body.input)


# ---------- agent6: Resume Screening Flow (router + or_) ----------
class ResumeState(FlowState):
    resume: str = ""
    job: str = ""
    score: int = 0


class ResumeFlow(Flow[ResumeState]):
    @start()
    def load_resumes(self):
        # input format: "RESUME: ... JOB DESCRIPTION: ..." (job part optional)
        parts = re.split(r"(?i)job\s*description\s*:", self.state.resume, maxsplit=1)
        self.state.resume = re.sub(r"(?i)^\s*resume\s*:", "", parts[0]).strip()
        self.state.job = parts[1].strip() if len(parts) > 1 else "a general professional role"

    @listen(load_resumes)
    def score_resume(self):
        r = ask(
            "Score how well this resume fits the job from 0 to 10. Reserve 10 for a perfect match "
            "with no meaningful gaps; deduct a point for each gap. Reply with only an integer.\n\n"
            f"JOB:\n{self.state.job}\n\nRESUME:\n{self.state.resume}"
        )
        m = re.search(r"\d+", r)
        self.state.score = min(10, int(m.group())) if m else 0

    @router(score_resume)
    def check_score(self):
        return "shortlist" if self.state.score >= 7 else "reject"

    @listen("shortlist")
    def summary_crew(self):
        s = self.state
        s.result = "SHORTLISTED\n\n" + mini_crew(
            "Recruiting Analyst", "Summarise a candidate's strengths and gaps for the role",
            f"Job:\n{s.job}\n\nResume:\n{s.resume}\n\nList strengths and gaps against the job.",
            "Two short bullet lists: Strengths, Gaps.",
        )

    @listen("reject")
    def draft_rejection(self):
        self.state.result = "REJECTED\n\n" + ask(
            "Write a short, polite, encouraging rejection email for a candidate. Sign off as "
            "'The Hiring Team' and use no [bracketed placeholders] (no company name, no your name). "
            f"Job:\n{self.state.job}\n\nResume:\n{self.state.resume[:1500]}"
        )

    @listen(or_(summary_crew, draft_rejection))
    def save_results(self):
        s = self.state
        log_row("results.csv", [s.score, s.result.split("\n")[0]])
        s.result = f"Score: {s.score}/10\n{s.result}"


@app.post("/agent6", response_model=Out)
def agent6(body: In = ex(
    "RESUME: Jane Doe. 5 years Python, FastAPI, AWS, led a team of 4. "
    "JOB DESCRIPTION: Senior Python backend engineer, FastAPI, cloud experience.")):
    return run(lambda t: run_flow(ResumeFlow, resume=t), body.input)


# ---------- agent7: Bug Triage Flow (router + Crews + or_) ----------
class IssueState(FlowState):
    issue: str = ""
    label: str = ""
    comment: str = ""


class TriageFlow(Flow[IssueState]):
    @start()
    def read_issue(self):
        self.state.issue = self.state.issue.strip()

    @listen(read_issue)
    def classify_issue(self):
        r = ask(
            f"Classify this issue. Reply with exactly one word: bug, feature or question.\n\n{self.state.issue}"
        ).lower()
        self.state.label = next((c for c in ("bug", "feature", "question") if c in r), "question")

    @router(classify_issue)
    def route_type(self):
        return self.state.label

    @listen("bug")
    def bug_crew(self):
        self.state.comment = mini_crew(
            "Bug Triager", "Assess severity and write clear reproduction steps",
            f"Issue:\n{self.state.issue}\n\nGive severity (low/medium/high/critical) with a reason, "
            "numbered steps to reproduce, and the likely cause.",
            "Severity, reproduction steps, likely cause.",
        )

    @listen("feature")
    def feature_crew(self):
        self.state.comment = mini_crew(
            "Product Manager", "Turn a feature request into a user story",
            f"Request:\n{self.state.issue}\n\nWrite a user story ('As a... I want... so that...') "
            "with 3 acceptance criteria.",
            "A user story with acceptance criteria.",
        )

    @listen("question")
    def answer_crew(self):
        self.state.comment = mini_crew(
            "Support Engineer", "Answer the user's question accurately",
            f"Question:\n{self.state.issue}\n\nIf the text contains a docs URL, scrape it and answer "
            "from it; otherwise search the web. Be concise and say if you are unsure.",
            "A short, helpful answer.", tools=[ScrapeWebsiteTool(), search],
        )

    @listen(or_(bug_crew, feature_crew, answer_crew))
    def post_reply(self):
        self.state.result = f"Label: {self.state.label}\n\n{self.state.comment}"


@app.post("/agent7", response_model=Out)
def agent7(body: In = ex(
    "App crashes with a NullPointerException every time I tap Save on the profile page "
    "after uploading a photo larger than 5MB. Android 14.")):
    return run(lambda t: run_flow(TriageFlow, issue=t), body.input)


# ---------- agent8: Content Repurposing Flow (parallel Crews + and_) ----------
class ArticleState(FlowState):
    article: str = ""
    points: str = ""
    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""


class RepurposeFlow(Flow[ArticleState]):
    @start()
    def load_article(self):
        a = self.state.article.strip()
        if re.fullmatch(r"https?://\S+", a):  # a bare URL -> fetch the article text
            a = str(ScrapeWebsiteTool().run(website_url=a))[:8000]
        self.state.article = a

    @listen(load_article)
    def extract_points(self):
        self.state.points = ask(
            f"Extract the 5 most important points from this article as bullets:\n\n{self.state.article[:8000]}"
        )

    @listen(extract_points)
    def linkedin_crew(self):
        self.state.linkedin = mini_crew(
            "LinkedIn Writer", "Write a LinkedIn post",
            f"Key points:\n{self.state.points}\n\nWrite a LinkedIn post: a strong hook line, then about 150 words.",
            "A LinkedIn post.",
        )

    @listen(extract_points)
    def twitter_crew(self):
        self.state.twitter = mini_crew(
            "Twitter Writer", "Write a tweet thread",
            f"Key points:\n{self.state.points}\n\nWrite a 6-tweet thread, numbered 1/6 to 6/6, each under 280 characters.",
            "A 6-tweet thread.",
        )

    @listen(extract_points)
    def newsletter_crew(self):
        self.state.newsletter = mini_crew(
            "Newsletter Writer", "Write a newsletter section",
            f"Key points:\n{self.state.points}\n\nWrite a short intro paragraph followed by 3 bullet takeaways.",
            "A newsletter section.",
        )

    @listen(and_(linkedin_crew, twitter_crew, newsletter_crew))
    def publish_pack(self):
        s = self.state
        save("linkedin.md", s.linkedin)
        save("twitter.md", s.twitter)
        save("newsletter.md", s.newsletter)
        s.result = f"## LinkedIn\n{s.linkedin}\n\n## Twitter thread\n{s.twitter}\n\n## Newsletter\n{s.newsletter}"


@app.post("/agent8", response_model=Out)
def agent8(body: In = ex("https://en.wikipedia.org/wiki/Remote_work")):
    return run(lambda t: run_flow(RepurposeFlow, article=t), body.input)


# ---------- agent9: Lead Qualification Flow (Crews + router + or_) ----------
class LeadState(FlowState):
    name: str = ""
    website: str = ""
    profile: str = ""
    score: str = ""
    email: str = ""


class LeadFlow(Flow[LeadState]):
    @start()
    def capture_lead(self):
        s = self.state
        s.website = re.search(r"https?://\S+", s.name).group()  # validated in agent9()
        s.name = re.sub(r"https?://\S+", "", s.name).strip(" ,-:") or "Unknown lead"

    @listen(capture_lead)
    def enrich_crew(self):
        self.state.profile = mini_crew(
            "Lead Researcher", "Profile a company from its website",
            f"Scrape {self.state.website} and summarise what the company does, its size signals "
            "and likely needs.",
            "A short company profile.", tools=[ScrapeWebsiteTool()],
        )

    @router(enrich_crew)
    def score_lead(self):
        r = ask(
            "Score this sales lead as hot, warm or cold (hot = clear fit and need). "
            f"Reply with one word.\n\nLead: {self.state.name}\nProfile:\n{self.state.profile}"
        ).lower()
        self.state.score = next((c for c in ("hot", "warm", "cold") if c in r), "cold")
        return self.state.score

    @listen("hot")
    def outreach_crew(self):
        self.state.email = mini_crew(
            "Sales Writer", "Draft a personal outreach email",
            f"Lead: {self.state.name}\nProfile:\n{self.state.profile}\n\nWrite a short personal email "
            "(under 120 words) with a subject line.",
            "An email with subject.",
        )

    @listen("warm")
    def add_to_nurture(self):
        self.state.email = "Added to the weekly tips nurture list. No email sent."

    @listen("cold")
    def archive_lead(self):
        self.state.email = "Archived. No action needed."

    @listen(or_(outreach_crew, add_to_nurture, archive_lead))
    def log_lead(self):
        s = self.state
        log_row("leads.csv", [s.name, s.website, s.score])
        s.result = f"Lead: {s.name} ({s.website})\nScore: {s.score}\n\nProfile:\n{s.profile}\n\nAction:\n{s.email}"


@app.post("/agent9", response_model=Out)
def agent9(body: In = ex("Stripe https://stripe.com")):
    def go(t: str) -> str:
        if not re.search(r"https?://\S+", t):
            raise HTTPException(400, "input must contain the lead's website URL, e.g. 'Acme Corp https://acme.com'")
        return run_flow(LeadFlow, name=t)

    return run(go, body.input)


# ---------- agent10: Approval Report Flow (Crews + router, automated review) ----------
class ReportState(FlowState):
    topic: str = ""
    research: str = ""
    draft: str = ""
    decision: str = ""
    feedback: str = ""


class ApprovalFlow(Flow[ReportState]):
    @start()
    def collect_topic(self):
        self.state.topic = self.state.topic.strip()

    @listen(collect_topic)
    def research_crew(self):
        m = llm()
        searcher = Agent(role="Researcher", goal="Find sources on the topic", backstory="A web researcher.", tools=[search], llm=m)
        checker = Agent(role="Fact Checker", goal="Drop weak or unsupported claims", backstory="A skeptical fact checker.", llm=m)
        summ = Agent(role="Research Summarizer", goal="Condense verified facts with sources", backstory="A concise analyst.", llm=m)
        t1 = Task(description=f"Search the web for: {self.state.topic}. Collect 6-8 facts with source links.", expected_output="Facts with links.", agent=searcher)
        t2 = Task(description="Remove any claim that lacks a source or looks doubtful.", expected_output="Verified facts with links.", agent=checker)
        t3 = Task(description="Summarise the verified facts as bullet notes with sources.", expected_output="Bullet research notes.", agent=summ)
        self.state.research = Crew(agents=[searcher, checker, summ], tasks=[t1, t2, t3], process=Process.sequential).kickoff().raw

    @listen(research_crew)
    def writing_crew(self):
        self.state.draft = mini_crew(
            "Report Writer", "Write a clear report from research notes",
            f"Topic: {self.state.topic}\n\nResearch notes:\n{self.state.research}\n\n"
            "Write a markdown report (about 400 words) with a title, sections and a Sources list.",
            "A markdown report.",
        )

    @listen(writing_crew)
    def review_draft(self):
        # Automated reviewer standing in for the human approval step (single stateless POST).
        r = ask(
            "You are a strict editor. Review this report for: clear structure, sources listed, "
            "no unsupported claims, no filler. Reply 'APPROVED' if it is publishable as-is, "
            "otherwise reply 'REVISE:' followed by specific fixes.\n\n" + self.state.draft
        )
        self.state.decision = "approved" if r.upper().startswith("APPROVED") else "revise"
        self.state.feedback = r

    @router(review_draft)
    def check_decision(self):
        return self.state.decision

    @listen("revise")
    def revision_crew(self):
        self.state.draft = mini_crew(
            "Revision Editor", "Apply review feedback to a report",
            f"Report:\n{self.state.draft}\n\nReviewer feedback:\n{self.state.feedback}\n\n"
            "Rewrite the full report applying every fix.",
            "The full revised markdown report.",
        )

    @listen(or_("approved", revision_crew))
    def publish_report(self):
        s = self.state
        save("final.md", s.draft)
        status = "approved" if s.decision == "approved" else "published after revision"
        s.result = f"Status: {status} (reviewed automatically)\n\n{s.draft}"


@app.post("/agent10", response_model=Out)
def agent10(body: In = ex("Benefits of a four-day work week")):
    return run(lambda t: run_flow(ApprovalFlow, topic=t), body.input)
