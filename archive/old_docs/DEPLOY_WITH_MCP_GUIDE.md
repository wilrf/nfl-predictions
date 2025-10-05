# **Deploy to GitHub While Keeping MCP.json Local**

## **The Solution: Keep Tokens Local, Push Template**

You can keep `.mcp.json` with real tokens on your computer and push a template to GitHub.

---

## **Strategy**

### **On Your Computer (Local)**
```
.mcp.json              ← Real tokens (stays local, never pushed)
```

### **On GitHub (Public)**
```
.mcp.json.example      ← Template with placeholders (safe to push)
.gitignore             ← Blocks .mcp.json from being pushed
```

---

## **How It Works**

### **1. Your .gitignore Already Has:**
```
.mcp.json
```
This prevents `.mcp.json` from being pushed to GitHub.

### **2. We Created .mcp.json.example:**
```json
{
  "mcpServers": {
    "supabase": {
      "env": {
        "SUPABASE_ACCESS_TOKEN": "YOUR_SUPABASE_TOKEN_HERE"
      }
    },
    "github": {
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE"
      }
    }
  }
}
```
This is a **template** with fake placeholders - safe to push.

### **3. Your Real .mcp.json Stays on Your Computer**
- Contains real tokens
- Never gets pushed (blocked by .gitignore)
- Claude Code features keep working ✅

---

## **The Problem: Old Git History**

Even though `.mcp.json` is in `.gitignore` NOW, it was committed in the PAST.

**Old commits still contain the real tokens.**

Git history is like a time machine - even if you delete a file today, it's still in yesterday's commits.

---

## **Solution: Fresh Start (Clean History)**

We'll remove old git history and start fresh:

### **What This Does:**
- ✅ Keeps ALL your files
- ✅ Keeps your real `.mcp.json` locally
- ✅ Removes old commits with exposed tokens
- ✅ Creates fresh git history
- ✅ Pushes template to GitHub

### **What You Keep:**
- ✅ All code
- ✅ All models
- ✅ All planning docs
- ✅ Your `.mcp.json` with real tokens (local only)
- ✅ Claude Code features working

### **What You Lose:**
- ❌ Old git commit history (but files stay)

---

## **Step-by-Step Process**

### **Step 1: Backup Your Real .mcp.json**
```bash
# Copy your real .mcp.json to safe location
cp ~/.config/claude/mcp.json ~/mcp.json.backup
# OR if it's in project root:
cp .mcp.json ~/mcp.json.backup
```

### **Step 2: Remove Old Git History**
```bash
# Remove old git history (keeps all files)
rm -rf .git

# Start fresh
git init

# Stage everything
git add .
```

**What gets added:**
- ✅ `.mcp.json.example` (template - safe)
- ✅ All your code
- ✅ All planning docs
- ✅ `.gitignore` (blocks real .mcp.json)

**What gets blocked:**
- ❌ `.mcp.json` (real tokens - blocked by .gitignore)
- ❌ `venv/` (blocked by .gitignore)
- ❌ `node_modules/` (blocked by .gitignore)

### **Step 3: Commit and Push**
```bash
# Create fresh commit
git commit -m "Complete NFL Prediction System

- Random Forest + XGBoost ensemble (76.7% accuracy)
- Docker + pytest setup complete
- Comprehensive planning documentation
- 18,746+ Supabase records
- Production ready

Setup: Copy .mcp.json.example to .mcp.json and add your tokens

🤖 Generated with Claude Code"

# Push to GitHub
git branch -M main
git remote add origin https://github.com/wilrf/Sports-Model.git
git push origin main --force
```

### **Step 4: Restore Your Real .mcp.json (if needed)**
```bash
# If .mcp.json got deleted (it shouldn't, but just in case)
cp ~/mcp.json.backup .mcp.json
# OR copy from Claude Code config
cp ~/.config/claude/mcp.json .mcp.json
```

---

## **Verification Checklist**

After pushing, verify:

```bash
# Check .mcp.json is NOT staged
git status
# Should NOT show .mcp.json

# Check .mcp.json exists locally
ls -la .mcp.json
# Should show the file (with real tokens)

# Check GitHub doesn't have real tokens
# Visit: https://github.com/wilrf/Sports-Model
# Should see .mcp.json.example (template)
# Should NOT see .mcp.json (real tokens)
```

---

## **For Other Developers (or Future You)**

When someone clones the repo:

```bash
# Clone repo
git clone https://github.com/wilrf/Sports-Model.git
cd Sports-Model

# Copy template
cp .mcp.json.example .mcp.json

# Edit with real tokens
nano .mcp.json
# Replace YOUR_GITHUB_TOKEN_HERE with actual token
# Replace YOUR_SUPABASE_TOKEN_HERE with actual token

# Now Claude Code works!
```

---

## **Why This is Safe**

### **Your Local Machine:**
```
Sports Model/
├── .mcp.json              ← Real tokens (blocked by .gitignore)
├── .mcp.json.example      ← Template (will be pushed)
└── .gitignore             ← Contains .mcp.json
```

**When you run `git add .`:**
- ✅ Adds `.mcp.json.example` (safe)
- ❌ Ignores `.mcp.json` (has real tokens)

### **On GitHub:**
```
Sports-Model/
├── .mcp.json.example      ← Template only (safe)
└── .gitignore             ← Contains .mcp.json
```

**Anyone who clones:**
- Gets template
- Must add their own tokens
- Your tokens stay private ✅

---

## **The Complete Flow**

```
┌─────────────────────────────────────────┐
│  YOUR COMPUTER (Local)                  │
│                                         │
│  .mcp.json ← Real tokens                │
│  (in .gitignore, never pushed)          │
│                                         │
│  Claude Code reads this ✅              │
│  Features work ✅                       │
└─────────────────────────────────────────┘
                    ↓
            git add . (ignores .mcp.json)
                    ↓
            git commit (no tokens)
                    ↓
            git push
                    ↓
┌─────────────────────────────────────────┐
│  GITHUB (Public)                        │
│                                         │
│  .mcp.json.example ← Template only      │
│  (safe placeholders)                    │
│                                         │
│  No real tokens ✅                      │
│  Safe to be public ✅                   │
└─────────────────────────────────────────┘
```

---

## **Commands to Run (All at Once)**

I'll run these for you:

```bash
# Backup (just in case)
cp .mcp.json ~/.mcp.json.backup 2>/dev/null || echo "No .mcp.json in current dir"

# Fresh git
rm -rf .git
git init

# Add everything (but .mcp.json is ignored)
git add .

# Commit
git commit -m "Complete NFL Prediction System

- Random Forest + XGBoost ensemble (76.7% accuracy)
- Docker + pytest setup complete
- Comprehensive planning documentation
- 18,746+ Supabase records
- Production ready

Setup: Copy .mcp.json.example to .mcp.json and add your tokens

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push
git branch -M main
git remote add origin https://github.com/wilrf/Sports-Model.git
git push origin main --force

# Verify .mcp.json still exists locally
ls -la .mcp.json
```

---

## **Summary**

✅ **Your real `.mcp.json` stays on your computer**
✅ **Claude Code features keep working**
✅ **GitHub gets template only (safe)**
✅ **No tokens exposed**
✅ **Ready to deploy to Vercel**

**Ready to run this?** Just say "yes" and I'll execute the commands.
