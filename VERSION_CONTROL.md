# Version Control Guide

This document extends AGENTS.md with version control and CI/CD instructions.

## Git/GitHub Setup

Repository: `https://github.com/wizrdfab/sofia`

### Git Workflow

1. **Before making changes**: Always pull latest
   ```bash
   git pull origin main
   ```

2. **After making changes**: Commit with descriptive message
   ```bash
   git add -A
   git commit -m "type(scope): brief description

   - Detail 1
   - Detail 2"
   ```

3. **Push changes**:
   ```bash
   git push origin main
   ```

### Commit Message Format

Use conventional commits format:
- `feat(module)`: New feature
- `fix(module)`: Bug fix
- `refactor(module)`: Code refactoring
- `docs`: Documentation changes
- `test`: Test changes
- `chore`: Maintenance tasks

Example:
```
feat(predictor): add confidence threshold parameter

- Added --min-confidence flag to predictor.py
- Default threshold set to 0.6
- Updated backtest to respect threshold
```

### Branching Strategy

- `main` - Production-ready code (auto-deploys to Azure VM)
- `develop` - Development branch for testing
- `feature/*` - Feature branches (merge to develop)
- `hotfix/*` - Emergency fixes (merge to main)

### Rolling Back Changes

To rollback to a previous version:

```bash
# View commit history
git log --oneline -20

# Rollback to specific commit (keeps history)
git revert <commit-hash>

# Hard reset to previous state (destructive)
git reset --hard <commit-hash>
git push --force origin main  # CAUTION: destroys remote history
```

To create a tagged version for easy reference:
```bash
git tag -a v1.5 -m "Version 1.5 - feature description"
git push origin v1.5
```

### Files NOT Tracked (see .gitignore)

Large data files are excluded from git:
- `data/` - Training data (~19GB)
- `data_collector/` - Data collection cache (~1.2GB)
- `models*/` - Trained model artifacts
- `logs*/` - Runtime logs
- `*.sqlite3` - Database files

See `DATA_SETUP.md` for how to sync data separately.

---

## CI/CD Pipeline (GitHub Actions)

The project has automated deployment to Azure VM via GitHub Actions.

### Available Workflows

1. **Deploy and Test** (`.github/workflows/deploy-and-test.yml`)
   - Triggers: Push to main/develop, PRs, manual
   - Actions: Lints code, deploys to Azure VM, runs tests
   - Results: Visible in GitHub Actions console

2. **Manual Command** (`.github/workflows/manual-command.yml`)
   - Run any command on Azure VM from GitHub UI
   - No SSH required - results in Actions console

3. **Sync Data** (`.github/workflows/sync-data.yml`)
   - Verify or sync large data files to Azure VM

### Viewing Test Results Without SSH

1. Go to GitHub repository > Actions tab
2. Click on the workflow run
3. Results appear in the job summary and logs
4. Artifacts (detailed logs) downloadable for 30 days

### Manual Deployment (Without CI/CD)

If GitHub Actions is unavailable, deploy manually:

```bash
# On Azure VM via SSH
cd ~/sofia
git pull origin main
pip install -r requirements.txt

# Run tests
python backtest.py --quick
```

### Required GitHub Secrets

Set these in GitHub repo Settings > Secrets:
- `AZURE_VM_HOST` - VM IP/hostname
- `AZURE_VM_USER` - SSH username
- `AZURE_VM_SSH_KEY` - Private SSH key
- `AZURE_VM_PATH` - Deployment path on VM

---

## Manual Operations Guide

For when AI agents are unavailable or for human developers.

### Running Training

```bash
# Basic training
python train.py

# With specific parameters
python train.py --symbol BTCUSDT --timeframe 5m

# Hyperparameter optimization
python hyperopt.py
```

### Running Backtest

```bash
# Quick backtest
python backtest.py --quick

# Full backtest with specific model
python backtest.py --model-dir models_test
```

### Running Live Trading (Paper)

```bash
# Simulated trading
python live_trading_funds_simulated.py

# With specific config
python live_trading.py --config config.py
```

### Common Fixes

**Import errors**: Check `requirements.txt` and reinstall
```bash
pip install -r requirements.txt
```

**Model not found**: Retrain or copy from backup
```bash
python train.py --output-dir models
```

**Data missing**: See `DATA_SETUP.md` for sync instructions

### IDE Setup (VS Code)

1. Open folder in VS Code
2. Install Python extension
3. Select interpreter: Python 3.10+
4. Recommended extensions:
   - Python
   - Pylance
   - GitLens

### File Structure Quick Reference

```
sofia/
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Training data (not in git)
├── models/               # Trained models (not in git)
├── *.py                  # Source code
├── AGENTS.md             # AI agent instructions
├── VERSION_CONTROL.md    # This file
├── DATA_SETUP.md         # Data sync guide
├── Manual.md             # User manual
├── requirements.txt      # Python dependencies
└── ai-log.txt           # AI change log
```
