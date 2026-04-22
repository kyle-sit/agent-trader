# agent-trader

News-driven market intelligence and paper-trading pipeline.

This project combines:
- multi-source news ingestion
- LLM-based market impact analysis
- technical analysis confirmation
- portfolio-aware trade execution on Alpaca paper
- separate portfolio rebalancing against hard risk constraints
- a learning layer for future signal recalibration

## What it does

High-level flow:

```text
NEWS DETECTION -> LLM ANALYSIS -> TA CONFIRMATION -> SIGNAL SCORING -> EXECUTION
```

Core ideas:
- **news-first** signal generation
- **TA for timing / confirmation**
- **GPT-routed reasoning** for market interpretation and execution decisions
- **hard portfolio risk rules** that block unsafe new trades
- **separate rebalance workflow** to trim existing positions back toward constraints

## Current architecture

### Main trading pipeline
The main live pipeline is:
- `scripts/signal_executor.py`

It runs:
1. source ingestion
2. headline analysis
3. ticker extraction
4. TA
5. news/TA correlation
6. alerts
7. paper-trade execution

### Separate rebalance workflow
The separate remediation script is:
- `scripts/rebalance_portfolio.py`

It:
1. inspects the current portfolio
2. checks current hard-rule violations
3. asks the LLM for the best trim/close plan
4. optionally executes those reductions

### Learning layer
The current learning script is:
- `scripts/learning_engine.py`

It tracks:
- graded trade outcomes
- topic / ticker / TA-pattern performance
- learned scoring multipliers for future signals

## Project structure

```text
agent-trader/
├── README.md
├── .env.example
├── .gitignore
├── data/
│   ├── llm_config.example.json
│   └── portfolio_risk_rules.example.json
└── scripts/
    ├── signal_executor.py
    ├── rebalance_portfolio.py
    ├── correlation_engine.py
    ├── news_analyzer.py
    ├── ta_engine.py
    ├── twitter_intel.py
    ├── portfolio_risk.py
    ├── position_manager.py
    ├── learning_engine.py
    ├── llm_router.py
    ├── llm_hooks.py
    └── supporting test / utility scripts
```

## Important files

### Runtime code
- `scripts/signal_executor.py` — main live paper-trading pipeline
- `scripts/rebalance_portfolio.py` — separate rebalance / trim execution
- `scripts/correlation_engine.py` — news + TA -> scored trade signals
- `scripts/news_analyzer.py` — source ingestion + LLM analysis
- `scripts/ta_engine.py` — indicator calculation / trade levels
- `scripts/twitter_intel.py` — Tier 1 X/Twitter ingestion with batched GPT relevance filtering
- `scripts/portfolio_risk.py` — risk rules, portfolio inspection, pre-trade checks
- `scripts/learning_engine.py` — graded outcomes and learned score adjustments
- `scripts/llm_router.py` — centralized LLM routing
- `scripts/llm_hooks.py` — LLM prompt wrappers

### Example config files
- `data/llm_config.example.json`
- `data/portfolio_risk_rules.example.json`

### Local runtime files (not committed)
The live system writes private/local state under `~/market-intel/data/`, such as:
- `trade_log.jsonl`
- `trade_outcomes.jsonl`
- `runtime_log.jsonl`
- `rebalance_log.jsonl`
- `position_state.json`
- `llm_config.json`
- `portfolio_risk_rules.json`
- `scoring_weights.json`
- `learned_patterns.json`

These are intentionally gitignored.

## Setup

### Python dependencies
Install the Python packages used by the pipeline:

```bash
pip3 install yfinance finnhub-python alpaca-trade-api ta trafilatura googlenewsdecoder python-dotenv anthropic
```

Additional tooling used by parts of the system:

```bash
go install github.com/Hyaxia/blogwatcher/cmd/blogwatcher@latest
npm install -g @steipete/bird
```

## Environment variables
Use `.env.example` as a reference.

Expected values include:

```bash
FINNHUB_API_KEY=
APCA_API_KEY_ID=
APCA_API_SECRET_KEY=
APCA_API_BASE_URL=https://paper-api.alpaca.markets
DISCORD_WEBHOOK_URL=
ANTHROPIC_API_KEY=
ANTHROPIC_TOKEN=
```

This project expects local credentials to live outside git, commonly in:
- `~/.hermes/.env`

## Local config
Create local runtime config files from the examples if needed:

```bash
cp data/llm_config.example.json data/llm_config.json
cp data/portfolio_risk_rules.example.json data/portfolio_risk_rules.json
```

## LLM routing
The project uses `scripts/llm_router.py` as the central LLM entrypoint.

Current design supports:
- GPT via Codex OAuth / ChatGPT backend
- optional Anthropic path
- optional Ollama path

In the current live setup this project is commonly run in **GPT-only mode** with:
- `disable_fallbacks: true`
- `critical -> gpt`
- `secondary -> gpt`

You can inspect the live router config with:

```bash
cd scripts
python3 llm_router.py --config
```

## X / Twitter ingestion
The current X pipeline uses:
- direct **Tier 1 monitored accounts only**
- **replies excluded**
- **30-minute freshness filter**
- **batched GPT relevance filtering** on retrieved tweets

This means only GPT-approved, recent, direct-account tweets are turned into market headlines.

## Portfolio hard rules
Risk rules are loaded from:
- example file: `data/portfolio_risk_rules.example.json`
- live runtime file: `data/portfolio_risk_rules.json`

Current example defaults:
- max open positions: 40
- max stock position: 5%
- max ETF position: 7%
- max sector exposure: 20%
- max theme exposure: 12%
- cash floor: 5%
- top-5 concentration cap: 35%
- max risk-at-stop per trade: 0.5%
- max aggregate open risk: 5%

## Running the system

### Full live paper-trading pipeline
```bash
cd scripts
python3 signal_executor.py
```

### Alerts-only (no paper trades)
```bash
cd scripts
python3 signal_executor.py --alerts-only
```

### Dry-run (heuristic smoke test)
```bash
cd scripts
python3 signal_executor.py --dry-run
```

### Show current paper positions
```bash
cd scripts
python3 signal_executor.py --positions
```

### Portfolio risk report
```bash
cd scripts
python3 portfolio_risk.py
```

### Inspect active risk rules
```bash
cd scripts
python3 portfolio_risk.py --rules
```

### Rebalance dry-run
```bash
cd scripts
python3 rebalance_portfolio.py --dry-run
```

### Execute rebalance
```bash
cd scripts
python3 rebalance_portfolio.py
```

### Learning report
```bash
cd scripts
python3 learning_engine.py --report
```

### Adjust learned weights from graded outcomes
```bash
cd scripts
python3 learning_engine.py --adjust-weights
```

## Execution behavior notes

### Action-based trade validation
The execution layer can request a structured action decision from GPT, such as:
- `open_new`
- `add_to_position`
- `hold_existing`
- `reduce_position`
- `reverse_position`
- `skip`

These decisions are still constrained by hard deterministic portfolio rules.

### Batched trade validation
Step 7 trade validation is batched into a **single GPT call** for candidate trades instead of one call per signal. This reduces:
- runtime
- GPT usage
- timeout risk

### Runtime logging
The main pipeline writes runtime metrics to:
- `data/runtime_log.jsonl`

This includes:
- total runtime
- step timings
- signal counts
- execution summary

### Rebalance logging
The rebalance script writes to:
- `data/rebalance_log.jsonl`

## Learning loop
The current learning loop is semi-manual.

It currently works like this:
1. executed trades are logged
2. closed trades can be graded
3. outcomes are aggregated into patterns
4. weights are adjusted in `scoring_weights.json`
5. future signal scoring uses those learned weights

Current learning mostly affects:
- topic adjustments
- TA pattern adjustments
- ticker adjustments
- alignment bonuses / penalties
- conviction weighting

## GitHub safety / secrets
This repo is intended to be public-safe.

### Not included
The following are intentionally not committed:
- live API keys
- local auth stores
- runtime logs
- live config files
- portfolio state

### Before using this repo
You must supply your own:
- Alpaca paper credentials
- Finnhub key
- any optional Discord webhook or Anthropic credentials
- local Codex/Hermes auth if using GPT routing through local auth stores

## Notes / caveats
- This is a **paper-trading** workflow, not live capital execution.
- Some scripts assume a local environment similar to the original development machine.
- Some historical utility/test scripts remain in `scripts/` and may be rougher than the main execution path.
- X/Twitter access depends on local auth/cookie tooling and is environment-specific.
- Market data quality and latency depend on the chosen sources.

## Suggested next improvements
- add portfolio overlap lock for cron execution
- automate the learning/grading loop
- add separate rebalance cron orchestration docs
- improve setup ergonomics and environment validation
- add unit tests around risk and execution decision logic

## License
Add your preferred license before broader public distribution.
