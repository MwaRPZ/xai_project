# How to Share This Repository

## Option 1: GitHub (Recommended)

### Step 1: Create a GitHub Repository
1. Go to https://github.com/new
2. Repository name: `XAI-Unified-Platform` (or your choice)
3. Description: "Unified XAI interface for deepfake audio and lung cancer detection"
4. Choose **Public** or **Private** (Private if you want to control access)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Push Your Code
```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/XAI-Unified-Platform.git

# Push your code
git push -u origin main
```

If you get an error about "master" vs "main":
```bash
git branch -M main
git push -u origin main
```

### Step 3: Share with Teammates
1. Go to your repository on GitHub
2. Click "Settings" → "Collaborators"
3. Click "Add people"
4. Enter your teammates' GitHub usernames or emails
5. They'll receive an invitation to collaborate

**Share this URL with them:**
```
https://github.com/YOUR_USERNAME/XAI-Unified-Platform
```

---

## Option 2: GitLab

### Step 1: Create a GitLab Project
1. Go to https://gitlab.com/projects/new
2. Project name: `XAI-Unified-Platform`
3. Visibility: Private or Public
4. **Uncheck** "Initialize repository with a README"
5. Click "Create project"

### Step 2: Push Your Code
```bash
git remote add origin https://gitlab.com/YOUR_USERNAME/XAI-Unified-Platform.git
git push -u origin main
```

### Step 3: Add Team Members
1. Go to Project → Members
2. Click "Invite members"
3. Add teammates by email or username
4. Set role (Developer or Maintainer)

---

## Option 3: Share via ZIP (No Git Required)

If teammates don't have Git:

### Create a Clean ZIP
```bash
# Windows PowerShell
Compress-Archive -Path * -DestinationPath XAI-Project.zip -Force
```

**Note**: This will include everything except what's in `.gitignore`

Share the ZIP via:
- Email
- Google Drive / OneDrive
- Slack / Teams

---

## What Your Teammates Will Do

Once they have access to the repository:

```bash
# 1. Clone the repository
git clone <repository-url>
cd XAI-Unified-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

---

## Repository Size Note

✅ **Your repository is lightweight** (~5-10 MB) because:
- Data files are excluded (`.gitignore`)
- Model weights are excluded
- Only code and documentation are included

❌ **Do NOT commit**:
- `data/` folder (datasets are huge)
- `models/*.pth` files (trained models)
- `.cache/` folder (Kaggle cache)

---

## Updating the Repository

When you make changes:

```bash
# Check what changed
git status

# Stage changes
git add .

# Commit with a message
git commit -m "Add feature: XYZ"

# Push to GitHub/GitLab
git push
```

Teammates can pull your updates:
```bash
git pull
```

---

## Troubleshooting

**Issue**: "Permission denied (publickey)"
- **Fix**: Set up SSH keys or use HTTPS URL

**Issue**: "Repository too large"
- **Fix**: Check `.gitignore` is working: `git ls-files | grep -E "\.pth|\.wav|data/"`

**Issue**: Merge conflicts
- **Fix**: Communicate with teammates about who's editing what files

---

## Quick Commands Reference

```bash
# See current status
git status

# Pull latest changes
git pull

# Create a new branch
git checkout -b feature/my-feature

# Switch branches
git checkout main

# See commit history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1
```
