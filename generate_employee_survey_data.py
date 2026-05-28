"""
generate_employee_survey_data.py

Generates a synthetic employee survey dataset with:
  - Realistic demographics (job_level, department, tenure)
  - Comments that vary in sentiment (positive, neutral, negative)
  - Comments organized by latent themes — useful for downstream
    topic modeling, thematic clustering, or LLM-based theme extraction

Themes baked in:
    1. management_leadership
    2. work_life_balance
    3. compensation_benefits
    4. growth_development
    5. team_collaboration
    6. culture_inclusion

Output: employee_survey_data.csv
"""

import random
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# pip install faker pandas
# If you don't have faker: pip install faker

random.seed(42)
fake = Faker()
Faker.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────

N = 300  # number of survey responses

DEPARTMENTS = [
    "Engineering", "Product", "Design", "Marketing",
    "Sales", "People & HR", "Finance", "Customer Success", "Legal"
]

JOB_LEVELS = ["IC1", "IC2", "IC3", "IC4", "IC5", "M1", "M2", "M3"]

# Probability that a given response leans positive / neutral / negative
SENTIMENT_WEIGHTS = [0.40, 0.25, 0.35]  # pos, neutral, neg

# ── Comment bank by theme × sentiment ─────────────────────────────────────────
# Each comment is a (text, theme, sentiment) tuple.
# Mix two themes in one comment to simulate real messiness.

COMMENT_BANK = {

    # ── management_leadership ────────────────────────────────────────────────
    ("management_leadership", "positive"): [
        "My manager is one of the best I've had — always clear on expectations and genuinely invested in my growth.",
        "Senior leadership does a great job communicating the company direction. I always feel like I know where we're headed.",
        "My skip-level is approachable and I feel comfortable raising concerns without fear.",
        "Leadership has been transparent about the challenges we're facing and I appreciate the honesty.",
        "My manager gives me autonomy while still being available when I need support. That balance is hard to find.",
        "I have a lot of trust in the direction leadership is taking. Decisions feel thoughtful rather than reactive.",
    ],
    ("management_leadership", "neutral"): [
        "Management style varies a lot by team. My experience has been fine but I've heard others struggle.",
        "Leadership communicates the big picture but the details often get lost by the time they reach my level.",
        "My manager is competent but not particularly invested in my development. I'm mostly self-directed.",
        "Senior leadership is fairly removed from day-to-day realities. Not a problem but worth noting.",
        "I've had three managers in two years, so it's hard to evaluate management consistently.",
    ],
    ("management_leadership", "negative"): [
        "There's a serious lack of psychological safety on my team. Feedback flows one direction — down.",
        "My manager regularly takes credit for the team's work and it's demoralizing.",
        "Leadership makes big promises during all-hands and then we never hear about them again.",
        "Direction changes constantly with no explanation. It's impossible to prioritize anything.",
        "My manager has never once asked me about my career goals. I feel completely invisible.",
        "There's a big gap between what leadership says about our culture and what actually happens.",
        "Decisions feel arbitrary and top-down. We're rarely consulted even when changes directly affect us.",
    ],

    # ── work_life_balance ────────────────────────────────────────────────────
    ("work_life_balance", "positive"): [
        "The flexible work policy is a genuine differentiator. I can structure my day around my life.",
        "I've never felt pressure to answer messages outside of work hours. The team respects boundaries.",
        "My workload is challenging but sustainable. I rarely feel overwhelmed.",
        "The company actually encourages people to take PTO and fully unplug. That's rare.",
        "Async-first culture means I can do deep work without constant interruptions.",
    ],
    ("work_life_balance", "neutral"): [
        "Work-life balance depends entirely on your team and what quarter it is. Officially it's fine.",
        "There are busy seasons where balance slips, but it tends to normalize. Not a dealbreaker.",
        "I manage my own boundaries pretty well, but I can see how it would be harder for others.",
        "The policy is good on paper but individual managers enforce it very differently.",
    ],
    ("work_life_balance", "negative"): [
        "I haven't taken a real vacation in 18 months because the workload never lets up.",
        "There's a culture of long hours that's never stated explicitly but is very much expected.",
        "I'm regularly pinged after 9pm by leadership and there's an implied expectation to respond immediately.",
        "Staffing shortages mean the remaining team is spread impossibly thin. Burnout is rampant.",
        "PTO exists in theory, but the actual workload makes it impossible to disconnect.",
        "I've raised concerns about my workload three times and nothing has changed.",
    ],

    # ── compensation_benefits ────────────────────────────────────────────────
    ("compensation_benefits", "positive"): [
        "The total comp package is genuinely competitive. I benchmarked it and I'm fairly paid.",
        "The equity refresh program shows the company wants to retain people long-term.",
        "Healthcare benefits are excellent — one of the best I've seen at any company.",
        "Annual reviews are meaningful and my comp has tracked my contributions.",
        "The 401k match and financial wellness perks are a real differentiator.",
    ],
    ("compensation_benefits", "neutral"): [
        "Pay is acceptable but not exceptional. I stay for other reasons.",
        "Comp feels about market rate. I'm not complaining but there's no premium for the pace we work at.",
        "Benefits are solid but not what sets this place apart. The work itself does.",
        "Raises exist but they're modest. Meaningful jumps only seem to happen with a promotion.",
    ],
    ("compensation_benefits", "negative"): [
        "I discovered I'm paid significantly less than peers doing the same work. That's been demoralizing.",
        "The pay process is completely opaque. I have no idea how decisions get made.",
        "We haven't had a meaningful compensation adjustment in two years despite strong company performance.",
        "My comp hasn't kept up with inflation and conversations about it go nowhere.",
        "The bonus structure looks good on paper but the targets are set so high they're rarely achievable.",
        "Equity was a selling point when I joined, but the timeline and vesting make it feel hollow now.",
    ],

    # ── growth_development ────────────────────────────────────────────────────
    ("growth_development", "positive"): [
        "I've been promoted twice in three years and feel like my contributions are recognized.",
        "The company invests in learning — I've used the development budget every year without friction.",
        "I get to work on hard, novel problems every week. My skills are growing faster here than anywhere else.",
        "My manager proactively identified a stretch opportunity for me and backed me through it.",
        "There's a clear leveling framework and I always know what I need to demonstrate to advance.",
        "Cross-functional projects have given me exposure I wouldn't have gotten at a more siloed company.",
    ],
    ("growth_development", "neutral"): [
        "Growth is possible here but it's largely self-driven. The company won't map it out for you.",
        "The learning budget exists but I've never felt strongly encouraged to use it.",
        "Career paths are clearer in some functions than others. Mine is ambiguous.",
        "I've grown in my role but I'm starting to wonder if there's a ceiling.",
    ],
    ("growth_development", "negative"): [
        "I've been in the same role for three years with no promotion conversation ever initiated.",
        "The leveling framework is inconsistently applied and it seems more about politics than performance.",
        "There are no real mentorship opportunities here. You're expected to figure it out on your own.",
        "Development conversations happen at review time and are forgotten immediately after.",
        "I've watched external hires fill senior roles that internal candidates were clearly ready for.",
        "The pace here doesn't allow time for learning. Everything is execution, always.",
    ],

    # ── team_collaboration ────────────────────────────────────────────────────
    ("team_collaboration", "positive"): [
        "My team is genuinely one of the most collaborative and supportive I've ever worked with.",
        "Cross-functional work flows well here. Other teams are responsive and easy to work with.",
        "Knowledge sharing is a real norm — people are generous with their time and expertise.",
        "We have strong rituals for alignment and rarely duplicate work or step on each other.",
        "People here celebrate each other's wins publicly and often. It creates real positive energy.",
    ],
    ("team_collaboration", "neutral"): [
        "Collaboration depends a lot on which team you're working with. Some are great, some are siloed.",
        "We work well within the team but cross-functional alignment can be slow and frustrating.",
        "Tooling and processes for collaboration have improved but there's still friction.",
        "Some teams are very collaborative, others are protective of their work. No consistent culture.",
    ],
    ("team_collaboration", "negative"): [
        "There's a real problem with silos. Teams hoard information and don't communicate until it's too late.",
        "Cross-team prioritization is a constant struggle. Nobody wants to deprioritize their own roadmap.",
        "Collaboration is performative. People nod in meetings and then do whatever they planned anyway.",
        "There's internal competition rather than cooperation. It doesn't serve us well.",
        "Decisions are made without consulting the people who will be most impacted by them.",
    ],

    # ── culture_inclusion ─────────────────────────────────────────────────────
    ("culture_inclusion", "positive"): [
        "This is the most inclusive team I've been on. I feel like I can bring my whole self to work.",
        "The company walks the walk on DEI — it shows up in hiring, promotions, and how leaders behave.",
        "People are genuinely curious and kind here. The culture is a real competitive advantage.",
        "I've seen the company course-correct quickly when something wasn't working. That takes maturity.",
        "Psychological safety is high on my team. We have real debates and nobody shuts people down.",
        "The values aren't just marketing copy — I see them reflected in how decisions get made.",
    ],
    ("culture_inclusion", "neutral"): [
        "The culture is good but starting to shift as the company grows. I hope it holds.",
        "There's a stated commitment to inclusion but I haven't seen it translate into structural changes.",
        "Culture varies significantly by team. Org-wide initiatives feel disconnected from day-to-day experience.",
        "The company is in a transitional phase and the culture feels unsettled. Not bad, just uncertain.",
    ],
    ("culture_inclusion", "negative"): [
        "There's a persistent pattern of underrepresented employees leaving and leadership doesn't address it.",
        "The inclusion language is polished but the lived experience is very different.",
        "Certain voices consistently dominate meetings and leadership hasn't done anything about it.",
        "A few high performers are allowed to behave badly because of their output. It's corrosive.",
        "I've raised a culture concern through the proper channels and received no follow-up.",
        "The culture has shifted noticeably since the last round of layoffs. Trust is low.",
    ],
}

# ── Multi-theme comment templates ─────────────────────────────────────────────
# These blend two themes, as real open-ends often do.

MULTI_THEME_TEMPLATES = {
    "positive": [
        ("management_leadership", "growth_development",
         "My manager is a genuine advocate for my development. She proactively nominated me for a high-visibility project and the growth has been real."),
        ("work_life_balance", "culture_inclusion",
         "The flexibility and the culture of respect here make this job sustainable in a way I haven't experienced before."),
        ("team_collaboration", "growth_development",
         "I learn so much just from working alongside my teammates. The collaboration here accelerates my development in a way that's hard to quantify."),
        ("compensation_benefits", "culture_inclusion",
         "The pay is fair and the culture backs it up — I feel valued in concrete ways, not just in words."),
        ("management_leadership", "culture_inclusion",
         "Leadership actively role-models the values they espouse. That consistency builds real trust over time."),
    ],
    "neutral": [
        ("management_leadership", "compensation_benefits",
         "My manager is fine and my comp is competitive, but I'm not inspired by the direction. Neither strongly positive nor negative, honestly."),
        ("work_life_balance", "growth_development",
         "Balance is manageable but growth opportunities are slow to materialize. I'd trade some balance for more challenge."),
        ("team_collaboration", "culture_inclusion",
         "My immediate team is great but the broader culture is harder to read. It depends a lot on who you work with."),
    ],
    "negative": [
        ("management_leadership", "work_life_balance",
         "My manager doesn't respect boundaries and the expectations for availability are unreasonable. It's affecting my health."),
        ("growth_development", "compensation_benefits",
         "No promotions, no real raises, and no clarity on what it would take to change that. The investment feels very one-directional."),
        ("culture_inclusion", "management_leadership",
         "Leadership talks about psychological safety but punishes people who speak up. The hypocrisy is demoralizing."),
        ("team_collaboration", "work_life_balance",
         "Poor cross-team coordination means everything is a fire drill. The constant urgency is exhausting."),
        ("growth_development", "culture_inclusion",
         "Advancement seems to go to people who are well-networked, not people who perform. It doesn't feel merit-based."),
    ],
}


# ── Helper functions ──────────────────────────────────────────────────────────

def pick_comment(sentiment: str) -> tuple[str, str, str]:
    """
    Returns (comment_text, primary_theme, sentiment).
    ~20% of the time draws from multi-theme templates.
    """
    if random.random() < 0.20:
        template = random.choice(MULTI_THEME_TEMPLATES[sentiment])
        theme_a, theme_b, text = template
        return text, f"{theme_a}+{theme_b}", sentiment
    else:
        theme = random.choice(list(set(k[0] for k in COMMENT_BANK)))
        options = COMMENT_BANK.get((theme, sentiment), [])
        if not options:
            theme = random.choice(list(set(k[0] for k in COMMENT_BANK)))
            options = COMMENT_BANK[(theme, sentiment)]
        return random.choice(options), theme, sentiment


def random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


# ── Generate dataset ──────────────────────────────────────────────────────────

records = []
survey_date = datetime(2024, 11, 1)

for i in range(1, N + 1):
    department = random.choice(DEPARTMENTS)
    job_level = random.choice(JOB_LEVELS)

    # Tenure: higher levels skew longer
    level_index = JOB_LEVELS.index(job_level)
    min_tenure_days = level_index * 180
    max_tenure_days = min_tenure_days + 365 * 4
    tenure_days = random.randint(min_tenure_days, max(min_tenure_days + 30, max_tenure_days))
    start_date = survey_date - timedelta(days=tenure_days)
    tenure_years = round(tenure_days / 365, 1)

    # Sentiment: slightly skew negative for more senior employees (burnout signal)
    weights = SENTIMENT_WEIGHTS.copy()
    if level_index >= 5:  # managers
        weights = [0.30, 0.20, 0.50]
    sentiment_label = random.choices(["positive", "neutral", "negative"], weights=weights)[0]

    comment_text, theme, sentiment = pick_comment(sentiment_label)

    records.append({
        "employee_id": f"EMP{i:04d}",
        "job_level": job_level,
        "department": department,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "survey_date": survey_date.strftime("%Y-%m-%d"),
        "tenure_years": tenure_years,
        "comment": comment_text,
        # Metadata columns — useful for evaluation / ground truth during development.
        # Strip these before running blind sentiment/theme analysis if you want a clean test.
        "_ground_truth_sentiment": sentiment,
        "_ground_truth_theme": theme,
    })

df = pd.DataFrame(records)

# ── Save ───────────────────────────────────────────────────────────────────────

output_path = "employee_survey_data.csv"
df.to_csv(output_path, index=False)

print(f"✓ Saved {len(df)} rows to {output_path}")
print()
print("── Column summary ─────────────────────────────────────────────────────")
print(df.dtypes.to_string())
print()
print("── Sentiment distribution ─────────────────────────────────────────────")
print(df["_ground_truth_sentiment"].value_counts().to_string())
print()
print("── Theme distribution ─────────────────────────────────────────────────")
print(df["_ground_truth_theme"].value_counts().to_string())
print()
print("── Sample rows ────────────────────────────────────────────────────────")
print(df[["employee_id", "job_level", "department", "tenure_years",
          "_ground_truth_sentiment", "_ground_truth_theme"]].sample(5).to_string(index=False))
print()
print("── Sample comments ────────────────────────────────────────────────────")
for _, row in df.sample(3, random_state=1).iterrows():
    print(f"[{row['_ground_truth_sentiment'].upper()} | {row['_ground_truth_theme']}]")
    print(f"  {row['comment']}")
    print()
