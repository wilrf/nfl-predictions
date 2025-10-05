# **Recommended MCPs & CLIs for NFL Prediction System**
## **Essential Tools Analysis**

**Date**: 2025-10-05
**Current Setup**: Python 3.9.6, basic tools (jq, curl, npm, redis-cli)

---

## **Executive Summary**

### **Currently Missing (High Priority)**
1. ✅ **Supabase MCP** - Already have it!
2. ✅ **GitHub MCP** - Already have it!
3. ✅ **Playwright MCP** - Already have it!
4. ✅ **Vercel MCP** - Already have it!
5. ❌ **Docker** - Not installed
6. ❌ **PostgreSQL CLI** - Not installed
7. ❌ **pytest** - Listed in requirements but not globally available
8. ❌ **Python package manager (poetry/pipenv)** - Not installed

### **Recommended Additions**
- **SQLite MCP** - Better database inspection
- **Filesystem MCP** - Enhanced file operations
- **AWS CLI** - If considering cloud deployment
- **Monitoring MCPs** - For production observability

---

## **Category 1: Database & Data Management**

### **1.1 SQLite MCP (HIGH PRIORITY)**

**Status**: Not currently available
**Why You Need It**:
- Your system uses SQLite (`database/nfl_suggestions.db`)
- Current access is through custom `db_manager.py`
- MCP would provide better inspection, queries, migrations

**What It Provides**:
```python
# Current way (manual):
from database.db_manager import DatabaseManager
db = DatabaseManager()
db.query("SELECT * FROM games WHERE season = 2024")

# With SQLite MCP (automated):
# Direct SQL access through MCP
# Schema inspection
# Migration management
# Query optimization suggestions
```

**Installation**:
```bash
# Add to .mcp.json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "path/to/nfl_suggestions.db"]
    }
  }
}
```

**Cost**: Free
**Setup Time**: 5 minutes
**Value**: HIGH - Direct database access without custom code

### **1.2 PostgreSQL CLI + MCP (MEDIUM PRIORITY)**

**Status**: Not installed
**Why You Might Need It**:
- If migrating from SQLite to PostgreSQL (for scale)
- Better concurrent access than SQLite
- Production deployment standard

**Current Use**: You're using Supabase (which IS PostgreSQL)
**Status**: You already have Supabase MCP! ✅

**Recommendation**: Skip standalone PostgreSQL, use Supabase MCP

---

## **Category 2: Development & Testing**

### **2.1 Docker (HIGH PRIORITY)**

**Status**: ❌ Not installed
**Why You Need It**:
- Consistent development environment
- Easy deployment
- Test different Python versions
- Run services locally (Redis, Postgres)

**Use Cases for Your Project**:
```dockerfile
# Dockerfile for NFL system
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY improved_nfl_system/ .

CMD ["python", "main.py"]
```

**Benefits**:
- Deploy to any cloud provider
- Consistent environment (dev = prod)
- Easy CI/CD setup
- Test Python 3.11+ without breaking local setup

**Installation**:
```bash
# macOS
brew install --cask docker

# Verify
docker --version
```

**Cost**: Free (Docker Desktop)
**Setup Time**: 15 minutes
**Value**: HIGH - Essential for modern development

### **2.2 pytest (MEDIUM PRIORITY)**

**Status**: In requirements.txt but not globally installed
**Why You Need It**:
- You have `tests/` directory
- Critical for Claude's plan (Day 6: Testing)
- Already listed in requirements

**Current Issue**: Not globally accessible

**Fix**:
```bash
# Install pytest globally
pip3 install pytest pytest-cov

# Or use in virtual environment (better)
cd improved_nfl_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Now pytest available
pytest tests/ -v
```

**Cost**: Free
**Setup Time**: 2 minutes
**Value**: HIGH - Required for testing

### **2.3 Poetry or Pipenv (MEDIUM PRIORITY)**

**Status**: Not installed
**Why You Need It**:
- Better dependency management than pip
- Lock files (reproducible installs)
- Virtual environment management
- Easier package publishing

**Current Issue**: Using pip + requirements.txt (basic)

**Poetry Example**:
```bash
# Install
curl -sSL https://install.python-poetry.org | python3 -

# Initialize in project
cd improved_nfl_system
poetry init

# Install dependencies
poetry add pandas numpy xgboost scikit-learn

# Create reproducible environment
poetry lock
poetry install
```

**Benefits**:
- `poetry.lock` ensures exact versions
- Easier dependency resolution
- Better than requirements.txt

**Cost**: Free
**Setup Time**: 10 minutes
**Value**: MEDIUM - Nice to have, not critical

---

## **Category 3: Deployment & Infrastructure**

### **3.1 Vercel CLI (LOW PRIORITY)**

**Status**: ✅ You have Vercel MCP already!
**Current Access**: Through MCP

**Optional CLI**:
```bash
# Install Vercel CLI for direct deployments
npm install -g vercel

# Deploy
cd improved_nfl_system/web
vercel deploy
```

**Recommendation**: MCP is sufficient, skip CLI unless doing manual deployments

### **3.2 AWS CLI (LOW PRIORITY)**

**Status**: Not installed
**Why You Might Need It**:
- If deploying to AWS (EC2, Lambda, S3)
- Current deployment: Vercel (serverless)

**When to install**:
- Only if moving from Vercel to AWS
- Need S3 for large model storage
- Want Lambda for predictions

**Cost**: Free CLI, AWS services cost money
**Setup Time**: 20 minutes + AWS account setup
**Value**: LOW - Only if changing deployment platform

### **3.3 Railway CLI or Fly.io CLI (MEDIUM PRIORITY)**

**Status**: Not installed
**Why You Might Need It**:
- Better Python deployment than Vercel
- Vercel is JavaScript-focused
- Railway/Fly.io are Python-friendly

**Railway Example**:
```bash
# Install
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up

# One command deploys entire Python app
```

**Benefits over Vercel**:
- Native Python support
- Easier deployment (FastAPI + workers)
- Better for background jobs

**Cost**: Free tier available
**Setup Time**: 15 minutes
**Value**: MEDIUM - Consider if Vercel deployment issues

---

## **Category 4: MCPs for Enhanced Development**

### **4.1 Filesystem MCP (MEDIUM PRIORITY)**

**Status**: Not installed
**Why You Might Need It**:
- Enhanced file operations
- Batch file processing
- Advanced search/replace

**Current Access**: Claude has Read/Write/Edit tools
**MCP Adds**:
- Bulk operations
- File watching
- Advanced patterns

**Value**: MEDIUM - Current tools sufficient for most cases

### **4.2 Time Series MCP (HIGH PRIORITY - If Exists)**

**Status**: Unknown if exists
**Why You'd Need It**:
- NFL data is time-series
- Walk-forward validation
- Temporal analysis

**What It Would Provide**:
```python
# Automatic time series validation
# Prevent data leakage
# Temporal cross-validation
# Seasonality detection
```

**Recommendation**: Check MCP marketplace for time-series tools

### **4.3 ML Model Registry MCP (MEDIUM PRIORITY)**

**Status**: Unknown if exists
**Why You'd Need It**:
- Track model versions
- Compare model performance
- Model lineage tracking

**Current Solution**: Custom versioning in `models/saved_models/`
**MCP Would Add**: Professional model management

**Alternatives**:
- MLflow (self-hosted)
- Weights & Biases (cloud service)
- DVC (data version control)

---

## **Category 5: Monitoring & Observability**

### **5.1 Sentry CLI (MEDIUM PRIORITY)**

**Status**: Not installed
**Why You Need It**:
- Error tracking in production
- Performance monitoring
- User session replay

**Use Case**:
```python
# Add to main.py
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0,
)

# Automatic error tracking
# No more digging through logs
```

**Cost**: Free tier (5k events/month)
**Setup Time**: 10 minutes
**Value**: MEDIUM - Important for production

### **5.2 DataDog or New Relic MCP (LOW PRIORITY)**

**Status**: Unknown if exists
**Why You Might Need It**:
- Production monitoring
- Performance metrics
- Log aggregation

**Cost**: Expensive ($15-100/mo)
**Recommendation**: Skip unless running production at scale

---

## **Category 6: Sports Data MCPs**

### **6.1 Sports Data MCP (HIGH VALUE IF EXISTS)**

**Status**: Check MCP marketplace
**What It Would Provide**:
- Direct access to sports APIs
- Odds data integration
- Real-time scores

**Current Solution**: Custom `odds_client.py`, `nfl_data_fetcher.py`

**MCP Would Replace**:
```python
# Current (custom code):
from data.odds_client import OddsClient
odds = OddsClient()
odds.get_odds("2024_01_KC_BAL")

# With MCP (automated):
# Direct API access through MCP
# Rate limiting handled
# Automatic retries
```

**Recommendation**: Check if exists, high value

### **6.2 The Odds API MCP (MEDIUM PRIORITY)**

**Status**: Likely doesn't exist (niche)
**DIY Option**: Create your own MCP

```javascript
// Custom MCP for The Odds API
// mcp-servers/odds-api/index.js

const server = new Server({
  name: "odds-api",
  version: "1.0.0",
}, {
  capabilities: {
    resources: {},
    tools: {},
  }
});

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{
    name: "get_odds",
    description: "Get odds for NFL games",
    inputSchema: { /* ... */ }
  }]
}));

// Export and use
```

**Value**: MEDIUM - Custom integration more flexible

---

## **Priority Installation List**

### **Install Today (High Priority)**

1. **Docker** - Essential for modern development
   ```bash
   brew install --cask docker
   ```

2. **pytest** (in virtual environment) - Required for testing
   ```bash
   cd improved_nfl_system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **SQLite MCP** - Better database access
   ```bash
   # Add to .mcp.json
   ```

### **Install This Week (Medium Priority)**

4. **Poetry** - Better dependency management
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

5. **Sentry** - Error tracking
   ```bash
   pip install sentry-sdk
   ```

6. **Railway CLI** - Better Python deployment
   ```bash
   npm install -g @railway/cli
   ```

### **Consider Later (Low Priority)**

7. **AWS CLI** - Only if migrating from Vercel
8. **Time Series MCP** - If it exists
9. **Custom Odds API MCP** - DIY project

---

## **Recommended .mcp.json Configuration**

```json
{
  "mcpServers": {
    "supabase": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-supabase"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token"
      }
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"]
    },
    "vercel": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-vercel"]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "/Users/wilfowler/Sports Model/improved_nfl_system/database/nfl_suggestions.db"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"]
    }
  }
}
```

---

## **Missing MCPs You Should Request**

### **1. Time Series Analysis MCP**
- Temporal validation
- Seasonality detection
- Trend analysis
- Walk-forward testing

### **2. ML Model Registry MCP**
- Version tracking
- Performance comparison
- A/B testing
- Model lineage

### **3. Sports Data Universal MCP**
- Multiple sports API integration
- Unified interface
- Rate limiting
- Caching

### **4. Betting Math MCP**
- Kelly criterion calculator
- Probability conversions
- Implied odds calculator
- CLV analysis

**Where to request**: MCP GitHub or Discord

---

## **Custom MCP Opportunities**

### **NFL Data MCP** (Build Your Own)

**What it would do**:
```python
# Unified interface to:
# - nfl_data_py
# - The Odds API
# - ESPN API
# - Weather API
# - Injury reports

# Single MCP for all NFL data
```

**Effort**: 1-2 weeks
**Value**: HIGH - Tailored to your needs

**Template**:
```javascript
// mcp-servers/nfl-data/index.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import axios from 'axios';

const server = new Server({
  name: "nfl-data",
  version: "1.0.0"
});

// Tool: Get game data
server.tool("get_game", async (params) => {
  // Fetch from multiple sources
  // Combine into single response
  return gameData;
});

// Tool: Get odds
server.tool("get_odds", async (params) => {
  // The Odds API integration
  return odds;
});

// Export
export default server;
```

**Benefits**:
- Single source of truth
- Consistent error handling
- Centralized rate limiting
- Easy testing

---

## **Tool Comparison Matrix**

| Tool | Priority | Cost | Setup Time | Value | Status |
|------|----------|------|------------|-------|--------|
| **Docker** | HIGH | Free | 15 min | HIGH | ❌ Install |
| **pytest** | HIGH | Free | 2 min | HIGH | ⚠️ Fix install |
| **SQLite MCP** | HIGH | Free | 5 min | HIGH | ❌ Add |
| **Supabase MCP** | N/A | Free | N/A | HIGH | ✅ Have |
| **GitHub MCP** | N/A | Free | N/A | MEDIUM | ✅ Have |
| **Playwright MCP** | N/A | Free | N/A | MEDIUM | ✅ Have |
| **Vercel MCP** | N/A | Free | N/A | MEDIUM | ✅ Have |
| **Poetry** | MEDIUM | Free | 10 min | MEDIUM | ❌ Install |
| **Railway CLI** | MEDIUM | Free | 15 min | MEDIUM | ❌ Optional |
| **Sentry** | MEDIUM | Free tier | 10 min | MEDIUM | ❌ Optional |
| **AWS CLI** | LOW | Free | 20 min | LOW | ❌ Skip |
| **Custom NFL MCP** | MEDIUM | Free | 1-2 weeks | HIGH | ❌ DIY project |

---

## **Immediate Action Plan**

### **Today (30 minutes)**
```bash
# 1. Install Docker (15 min)
brew install --cask docker

# 2. Fix pytest (2 min)
cd "improved_nfl_system"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Test pytest works
pytest tests/ -v

# 4. Add SQLite MCP (5 min)
# Create .mcp.json in project root
cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "improved_nfl_system/database/nfl_suggestions.db"
      ]
    }
  }
}
EOF
```

### **This Week (1 hour)**
```bash
# 5. Install Poetry (10 min)
curl -sSL https://install.python-poetry.org | python3 -

# 6. Migrate to Poetry (20 min)
cd improved_nfl_system
poetry init
poetry add pandas numpy xgboost scikit-learn fastapi uvicorn
poetry lock

# 7. Setup Sentry (15 min)
poetry add sentry-sdk
# Configure in main.py

# 8. Test Railway deployment (15 min)
npm install -g @railway/cli
railway login
railway init
```

### **This Month (Optional)**
- Build custom NFL Data MCP (1-2 weeks)
- Explore time series MCPs
- Setup monitoring dashboard

---

## **Cost Summary**

| Tool | Monthly Cost | Annual Cost |
|------|-------------|-------------|
| Docker | $0 | $0 |
| pytest | $0 | $0 |
| Poetry | $0 | $0 |
| MCPs (all) | $0 | $0 |
| Sentry (free tier) | $0 | $0 |
| Railway (free tier) | $0 | $0 |
| **TOTAL** | **$0** | **$0** |

**All recommended tools are FREE** (free tier or open source)

---

## **ROI Analysis**

### **Docker**
- **Cost**: $0
- **Time Saved**: 10 hours/year (environment issues)
- **Value**: Consistent deployments

### **pytest + Coverage**
- **Cost**: $0
- **Time Saved**: 20 hours/year (manual testing)
- **Value**: Catch bugs early

### **SQLite MCP**
- **Cost**: $0
- **Time Saved**: 5 hours/year (database debugging)
- **Value**: Faster development

### **Poetry**
- **Cost**: $0
- **Time Saved**: 8 hours/year (dependency conflicts)
- **Value**: Reproducible installs

### **Sentry**
- **Cost**: $0 (free tier)
- **Time Saved**: 15 hours/year (debugging production errors)
- **Value**: Find bugs users don't report

**Total Time Saved**: ~58 hours/year
**Total Cost**: $0/year
**ROI**: Infinite

---

## **Final Recommendations**

### **Must Install (Do Today)**
1. ✅ Docker
2. ✅ pytest (fix venv)
3. ✅ SQLite MCP

### **Should Install (This Week)**
4. ✅ Poetry
5. ✅ Sentry
6. ✅ Railway CLI

### **Nice to Have (This Month)**
7. ⚠️ Custom NFL Data MCP (DIY)
8. ⚠️ Filesystem MCP
9. ⚠️ Time Series MCP (if exists)

### **Skip (Not Needed)**
- ❌ AWS CLI (using Vercel/Railway)
- ❌ Standalone PostgreSQL (have Supabase)
- ❌ DataDog/New Relic (too expensive)
- ❌ Vercel CLI (have MCP)

---

## **Next Steps**

Run this command to get started:

```bash
# Complete setup script (5 minutes)
#!/bin/bash

echo "Installing essential tools for NFL prediction system..."

# 1. Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    brew install --cask docker
fi

# 2. pytest in venv
echo "Setting up Python environment..."
cd "improved_nfl_system"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. SQLite MCP
echo "Configuring SQLite MCP..."
cat > ../.mcp.json << 'EOF'
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "improved_nfl_system/database/nfl_suggestions.db"]
    }
  }
}
EOF

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Restart Claude Code to load SQLite MCP"
echo "2. Test: pytest tests/ -v"
echo "3. Test: docker --version"
```

Save as `setup_tools.sh` and run:
```bash
chmod +x setup_tools.sh
./setup_tools.sh
```
