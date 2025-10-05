# Archive

This folder contains old, obsolete, or duplicate files that are no longer needed for the active project.

## Structure

### `old_docs/`
Session documentation and setup guides that are no longer current:
- CLAIMS_REVIEW.md - Old system health review
- CLARIFICATION.md - Old token clarification doc
- DEPLOY_WITH_MCP_GUIDE.md - Old MCP deployment guide
- DOCKER_INSTALLATION.md - Docker setup instructions
- SETUP_COMPLETE.md - Old setup completion doc

### `planning_docs/`
Planning documents and implementation proposals:
- THEPland - Original 12-month implementation plan (rejected)
- TECHNICAL_IMPLEMENTATION_PLAN.md - Old technical plan
- theplan/ - Competing implementation plans folder

### `old_web_experiments/`
Deprecated web interface attempts:
- public/ - Old static HTML/CSS/JS attempt
- web_app/ - Old FastAPI web app (replaced by api/)

### `old_data_processing/`
Legacy data processing scripts (replaced by src/ modules):
- data_integration/ - Old data import scripts
- feature_engineering/ - Old feature builder scripts

### `duplicate_files/`
Empty or duplicate folders:
- api_endpoints/ - Empty folder
- vercel.json.old - Old Vercel config

### `temp_files/`
Temporary configuration files:
- playwright.config.py - Old Playwright test config

## Active Project Structure

The active project now has a clean structure:
- `src/` - Core NFL prediction system
- `web/` - Next.js frontend (production)
- `api/` - Vercel serverless functions
- `database/` - Database files
- `scripts/` - Utility scripts
- `saved_models/` - ML models
- `tests/` - Test files
- `docs/` - Current documentation
- `validation/` - Validation framework

## Restoration

If you need to restore any archived files:
```bash
# Example: Restore planning docs
cp archive/planning_docs/THEPland ./
```

**Note:** Most archived files are obsolete and should not be restored.
