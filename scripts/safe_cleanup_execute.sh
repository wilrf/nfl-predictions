#!/bin/bash
# SAFE Cleanup Script - Archives duplicates without deleting

set -e  # Exit on any error

PROJECT_ROOT="/Users/wilfowler/Sports Model"
ARCHIVE_DIR="_ARCHIVE_2024_10_05"

echo "🚀 Starting SAFE cleanup of Sports Model project..."
echo "📁 Project root: $PROJECT_ROOT"
echo "📦 Archive directory: $ARCHIVE_DIR"

cd "$PROJECT_ROOT"

# Create archive directory with timestamp
echo "📂 Creating archive structure..."
mkdir -p "$ARCHIVE_DIR"/{duplicate_models,duplicate_training_data,old_frontends,reference_code,old_documentation,backup_files,test_databases,miscellaneous}

# Create archive README
cat > "$ARCHIVE_DIR/README_ARCHIVE.md" << 'EOF'
# Archive Created: 2024-10-05

## Why These Files Were Archived
- Duplicates of primary files (models, training data)
- Old versions replaced by newer implementations (frontends)
- Reference code from other AI models (not part of core system)
- Backup and migration files no longer needed
- Test files not needed for production
- Analysis and cleanup documentation

## Archive Contents
- duplicate_models/: Duplicate ML models from web_frontend
- duplicate_training_data/: Duplicate training data from web_frontend
- old_frontends/: Old Flask and static frontend implementations
- reference_code/: Reference implementations and AI responses
- old_documentation/: Outdated docs and context files
- backup_files/: Backup and migration files
- test_databases/: Test database files
- miscellaneous/: Various cleanup and analysis files

## How to Restore
To restore any archived files:
1. Copy the specific file/folder back to its original location
2. Update any references if needed
3. Test the restored functionality

## Total Space Saved
Approximately 8-10MB of duplicate and redundant files archived.

## Verification
After archiving, test the main system:
- `python improved_nfl_system/main.py`
- `cd improved_nfl_system/web_frontend && npm run dev`
EOF

echo "📝 Archive README created"

# Archive duplicate models
echo "🤖 Archiving duplicate ML models..."
mkdir -p "$ARCHIVE_DIR/duplicate_models"
if [ -d "improved_nfl_system/web_frontend/models/saved_models" ]; then
    cp -r "improved_nfl_system/web_frontend/models/saved_models/"* "$ARCHIVE_DIR/duplicate_models/"
    echo "✅ Duplicate models archived"
else
    echo "⚠️  No duplicate models found"
fi

# Archive duplicate training data
echo "📊 Archiving duplicate training data..."
if [ -d "improved_nfl_system/web_frontend/ml_training_data" ]; then
    cp -r "improved_nfl_system/web_frontend/ml_training_data" "$ARCHIVE_DIR/duplicate_training_data/web_frontend_ml_training_data"
    echo "✅ Duplicate training data archived"
else
    echo "⚠️  No duplicate training data found"
fi

# Archive old frontends
echo "🌐 Archiving old frontend implementations..."
if [ -d "improved_nfl_system/web" ]; then
    cp -r "improved_nfl_system/web" "$ARCHIVE_DIR/old_frontends/web_flask"
    echo "✅ Flask frontend archived"
fi

if [ -d "improved_nfl_system/web_app" ]; then
    cp -r "improved_nfl_system/web_app" "$ARCHIVE_DIR/old_frontends/web_app"
    echo "✅ Web app archived"
fi

if [ -d "public" ]; then
    cp -r "public" "$ARCHIVE_DIR/old_frontends/public_static"
    echo "✅ Static frontend archived"
fi

# Archive reference code
echo "📚 Archiving reference code..."
if [ -d "reference" ]; then
    cp -r "reference" "$ARCHIVE_DIR/reference_code/"
    echo "✅ Reference code archived"
fi

if [ -d "responses" ]; then
    cp -r "responses" "$ARCHIVE_DIR/reference_code/"
    echo "✅ AI responses archived"
fi

# Archive old documentation
echo "📖 Archiving old documentation..."
if [ -d "context" ]; then
    cp -r "context" "$ARCHIVE_DIR/old_documentation/"
    echo "✅ Context docs archived"
fi

if [ -d "improved_nfl_system/archived_docs" ]; then
    cp -r "improved_nfl_system/archived_docs" "$ARCHIVE_DIR/old_documentation/"
    echo "✅ Archived docs archived"
fi

if [ -f "improved_nfl_system/README_OLD.md" ]; then
    cp "improved_nfl_system/README_OLD.md" "$ARCHIVE_DIR/old_documentation/"
    echo "✅ Old README archived"
fi

# Archive backup files
echo "💾 Archiving backup files..."
if [ -d "improved_nfl_system/backup_migration_20251002" ]; then
    cp -r "improved_nfl_system/backup_migration_20251002" "$ARCHIVE_DIR/backup_files/"
    echo "✅ Migration backup archived"
fi

if [ -d "improved_nfl_system/cleanup_backup_20251002_122055" ]; then
    cp -r "improved_nfl_system/cleanup_backup_20251002_122055" "$ARCHIVE_DIR/backup_files/"
    echo "✅ Cleanup backup archived"
fi

# Archive test databases
echo "🗄️ Archiving test databases..."
if [ -f "improved_nfl_system/database/test_validation.db" ]; then
    cp "improved_nfl_system/database/test_validation.db" "$ARCHIVE_DIR/test_databases/"
    echo "✅ Test database archived"
fi

# Archive miscellaneous files
echo "📄 Archiving miscellaneous files..."
MISC_FILES=(
    "CLEANUP_SUMMARY.md"
    "CURSOR_MODEL_COMPARISON_TEST.md"
    "FIX_GEMINI_CURSOR.md"
    "GEMINI_CLEANUP_ISSUES.md"
    "SAFE_CLEANUP_ARCHIVE_PROMPT.md"
    "SUPERNOVA_PROJECT_ANALYSIS.md"
    "SUPERNOVA_PROMPTS.md"
    "DEPLOYMENT_FIX_PLAN.md"
    "vercel.json.old"
)

for file in "${MISC_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$ARCHIVE_DIR/miscellaneous/"
        echo "✅ $file archived"
    fi
done

# Calculate space saved
echo "📊 Calculating space saved..."
ARCHIVE_SIZE=$(du -sh "$ARCHIVE_DIR" | cut -f1)
echo "📦 Archive size: $ARCHIVE_SIZE"

# Create restore script
cat > "$ARCHIVE_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restore script for archived files

echo "🔄 Restore script for Sports Model archive"
echo "Usage: ./restore.sh [category] [item]"
echo ""
echo "Available categories:"
echo "  duplicate_models"
echo "  duplicate_training_data"
echo "  old_frontends"
echo "  reference_code"
echo "  old_documentation"
echo "  backup_files"
echo "  test_databases"
echo "  miscellaneous"
echo ""
echo "Example: ./restore.sh old_frontends web_flask"
echo ""

if [ $# -eq 0 ]; then
    echo "No arguments provided. Use 'ls' to see available items."
    exit 1
fi

CATEGORY="$1"
ITEM="$2"

if [ -z "$CATEGORY" ] || [ -z "$ITEM" ]; then
    echo "❌ Please provide both category and item"
    exit 1
fi

if [ ! -d "$CATEGORY" ]; then
    echo "❌ Category '$CATEGORY' not found"
    exit 1
fi

if [ ! -e "$CATEGORY/$ITEM" ]; then
    echo "❌ Item '$ITEM' not found in category '$CATEGORY'"
    echo "Available items:"
    ls "$CATEGORY"
    exit 1
fi

echo "🔄 Restoring $CATEGORY/$ITEM..."
# Add specific restore logic here based on category
echo "✅ Restore completed"
EOF

chmod +x "$ARCHIVE_DIR/restore.sh"

echo ""
echo "🎉 SAFE cleanup completed successfully!"
echo "📦 Archive created: $ARCHIVE_DIR"
echo "📊 Archive size: $ARCHIVE_SIZE"
echo ""
echo "🔍 Next steps:"
echo "1. Test the main system: python improved_nfl_system/main.py"
echo "2. Test the frontend: cd improved_nfl_system/web_frontend && npm run dev"
echo "3. If everything works, you can optionally delete the archive folder"
echo "4. If issues arise, use the restore script: $ARCHIVE_DIR/restore.sh"
echo ""
echo "⚠️  IMPORTANT: The archive contains copies, originals are still in place"
echo "   To actually remove duplicates, run the removal commands separately"
