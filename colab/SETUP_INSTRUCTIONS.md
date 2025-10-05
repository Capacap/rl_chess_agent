# GitHub Authentication for Colab

## Problem
The repository `Capacap/rl_chess_agent` is private, so Colab cannot clone it without authentication.

## Solution Options

### Option 1: Make Repository Public (Recommended)

**Simplest approach - no authentication needed in Colab**

1. Go to https://github.com/Capacap/rl_chess_agent/settings
2. Scroll to bottom → "Danger Zone"
3. Click "Change visibility" → "Make public"
4. Confirm

**Then in Colab:**
```python
!git clone https://github.com/Capacap/rl_chess_agent.git
```

**Pros:**
- No authentication setup needed
- Works immediately
- Simpler workflow

**Cons:**
- Code is publicly visible (usually fine for academic projects)

---

### Option 2: Use Personal Access Token (Keep Private)

**If you need to keep the repo private:**

#### Step 1: Create GitHub Personal Access Token

1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Set:
   - Note: "Colab Training"
   - Expiration: 30 days
   - Scopes: Check ✓ `repo` (full control)
4. Click "Generate token"
5. **Copy token immediately** (you won't see it again)

#### Step 2: Update Colab Notebook

Replace the clone cell in `setup_test.ipynb` and `train.ipynb`:

```python
# Store token (Colab will prompt you to enter it)
from google.colab import userdata
import os

# First time: This will prompt for token input
# Subsequent runs: Retrieves stored token
try:
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
except:
    # Manually input token
    import getpass
    GITHUB_TOKEN = getpass.getpass('Enter GitHub token: ')

# Clone with authentication
!git clone https://{GITHUB_TOKEN}@github.com/Capacap/rl_chess_agent.git
%cd rl_chess_agent
```

**Or use Colab Secrets (more secure):**

1. In Colab, click 🔑 (key icon) in left sidebar
2. Add new secret:
   - Name: `GITHUB_TOKEN`
   - Value: (paste your token)
3. Enable "Notebook access"

Then in notebook:
```python
from google.colab import userdata
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
!git clone https://{GITHUB_TOKEN}@github.com/Capacap/rl_chess_agent.git
%cd rl_chess_agent
```

---

### Option 3: Upload Code Manually

**Least convenient but works:**

1. On local machine: `git archive --format=zip HEAD > rl_chess_agent.zip`
2. Upload zip to Google Drive
3. In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/rl_chess_agent.zip .
!unzip -q rl_chess_agent.zip -d rl_chess_agent
%cd rl_chess_agent
```

**Downside:** Must re-upload on every code change

---

## Recommendation

**For this project:** Make repository public (Option 1)

**Why:**
- Chess agent training is typically not sensitive
- Simplifies workflow significantly
- Standard for academic/competition projects
- Can make private again after Oct 22 if desired

**If keeping private is required:** Use Option 2 with Colab Secrets (most secure)
