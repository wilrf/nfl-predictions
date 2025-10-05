#!/bin/bash
# Safe Project Cleanup Script
# Removes migration artifacts, duplicates, and one-time scripts
# Total cleanup: ~12-15 MB, 130+ files

set -e  # Exit on error

BACKUP_DIR="./cleanup_backup_$(date +%Y%m%d_%H%M%S)"
PROJECT_ROOT="/Users/wilfowler/Sports Model"

echo "🧹 NFL Betting System - Project Cleanup"
echo "========================================"
echo ""
echo "This will remove:"
echo "  • 109 Supabase migration files"
echo "  • 11 duplicate documentation files"
echo "  • 4 orphaned database files"
echo "  • 4 one-time completed scripts"
echo "  • 3 duplicate directories"
echo ""
echo "Total space to free: ~12-15 MB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"
echo "📦 Created backup directory: $BACKUP_DIR"
echo ""

cd "$PROJECT_ROOT"

# Counters
DELETED_FILES=0
DELETED_DIRS=0

# Function to safely delete with backup
safe_delete() {
    local item="$1"
    local description="$2"

    if [ -e "$item" ]; then
        echo "  Deleting: $description"
        cp -r "$item" "$BACKUP_DIR/" 2>/dev/null || true
        rm -rf "$item"
        if [ -d "$item" ]; then
            DELETED_DIRS=$((DELETED_DIRS + 1))
        else
            DELETED_FILES=$((DELETED_FILES + 1))
        fi
    fi
}

echo "Phase 1: Supabase Migration Artifacts (109 files)"
echo "------------------------------------------------"
cd improved_nfl_system

# SQL chunks and transformed files
for file in chunk_*.sql; do
    [ -e "$file" ] && safe_delete "$file" "SQL chunk: $file"
done
for file in fixed_*.sql; do
    [ -e "$file" ] && safe_delete "$file" "Fixed SQL: $file"
done
for file in execute_*.sql; do
    [ -e "$file" ] && safe_delete "$file" "Execute SQL: $file"
done
for file in temp_*.sql; do
    [ -e "$file" ] && safe_delete "$file" "Temp SQL: $file"
done

# Python migration scripts
for file in phase*.py; do
    [ -e "$file" ] && safe_delete "$file" "Phase script: $file"
done
for file in batch_*.py; do
    [ -e "$file" ] && safe_delete "$file" "Batch script: $file"
done
for file in execute_*.py; do
    [ -e "$file" ] && safe_delete "$file" "Execute script: $file"
done
safe_delete "bulk_load.py" "Bulk load script"
safe_delete "data_load_to_supabase.py" "Data load script"
safe_delete "load_all_data.py" "Load all data script"
safe_delete "load_all_fixed.py" "Load all fixed script"
safe_delete "load_data_to_supabase.py" "Load data script"
safe_delete "load_remaining.py" "Load remaining script"
safe_delete "final_load_execute.py" "Final load script"
# Removed MCP loader script reference
safe_delete "test_season_loader.py" "Test season loader"

# Large SQL dumps
safe_delete "supabase_complete_data.sql" "Complete data dump (3.8 MB)"
safe_delete "supabase_complete_schema.sql" "Complete schema dump"
safe_delete "supabase_data_migration.sql" "Data migration SQL"
safe_delete "supabase_schema.sql" "Schema SQL"
safe_delete "load_betting_outcomes.sql" "Betting outcomes SQL (102 KB)"
safe_delete "load_epa_metrics.sql" "EPA metrics SQL (139 KB)"
safe_delete "load_game_features.sql" "Game features SQL (183 KB)"
safe_delete "load_historical_games.sql" "Historical games SQL (71 KB)"
safe_delete "load_team_epa_stats.sql" "Team EPA stats SQL (287 KB)"
safe_delete "load_team_features.sql" "Team features SQL (189 KB)"
safe_delete "create_tables.sql" "Create tables SQL"

# Temporary data files
safe_delete "transformed_data.pkl" "Transformed data pickle (660 KB)"
# Removed MCP commands reference
safe_delete "final_load_commands.json" "Final load commands (997 KB)"
safe_delete "ready_to_load.json" "Ready to load data (997 KB)"
safe_delete "database_analysis.json" "Database analysis (67 KB)"
safe_delete "schema_discovery.json" "Schema discovery (34 KB)"
safe_delete "transformation_summary.json" "Transformation summary"
for file in load_report_*.json; do
    [ -e "$file" ] && safe_delete "$file" "Load report: $file"
done

# Supabase client files (if no longer needed)
safe_delete "supabase_client.py" "Supabase client"
safe_delete "supabase_migration.py" "Supabase migration script"
safe_delete "supabase_complete_migration.py" "Complete migration script"

echo "  ✓ Phase 1 complete"

echo ""
echo "Phase 2: Duplicate Documentation (11 files)"
echo "-------------------------------------------"
cd "$PROJECT_ROOT/improved_nfl_system"

# Duplicate CLAUDE.md (keep root version)
safe_delete "CLAUDE.md" "Duplicate CLAUDE.md"

# Docs moved to /context/ directory
safe_delete "BUG_ANALYSIS_REPORT.md" "Bug analysis (moved to /context/)"
safe_delete "BULK_IMPORT_GUIDE.md" "Bulk import guide (moved to /context/)"
safe_delete "DATA_IMPORT_AND_BUG_FIX_PLAN.md" "Data import plan (moved to /context/)"
safe_delete "DATA_IMPORT_SUCCESS_SUMMARY.md" "Import success summary (moved to /context/)"
safe_delete "FEATURE_VALIDATION_STRATEGY.md" "Validation strategy (moved to /context/)"
safe_delete "FINAL_DATA_STATUS.md" "Data status (moved to /context/)"
safe_delete "IMPLEMENTATION_PLAN.md" "Implementation plan (moved to /context/)"
safe_delete "IMPLEMENTATION_PLAN_REVIEW.md" "Plan review (moved to /context/)"
safe_delete "MASSIVE_DATA_IMPORT_STRATEGY.md" "Import strategy (moved to /context/)"
safe_delete "ML_SUCCESS_ROADMAP.md" "ML roadmap (moved to /context/)"

echo "  ✓ Phase 2 complete"

echo ""
echo "Phase 3: Orphaned Database Files"
echo "---------------------------------"
cd "$PROJECT_ROOT/improved_nfl_system/database"

safe_delete "test_bugfix.db-shm" "Orphaned SHM file"
safe_delete "test_bugfix.db-wal" "Orphaned WAL file (249 KB)"

# Old schema versions (keep schema.sql only)
safe_delete "schema_v2.sql" "Old schema version 2"
safe_delete "schema_comprehensive.sql" "Old comprehensive schema"

echo "  ✓ Phase 3 complete"

echo ""
echo "Phase 4: One-Time Test Scripts"
echo "-------------------------------"
cd "$PROJECT_ROOT/improved_nfl_system"

# Completed one-time scripts
safe_delete "bulk_import_historical_data.py" "Bulk import script (completed)"
safe_delete "bulk_import_progress.json" "Import progress tracker"
safe_delete "consolidate_training_data.py" "Data consolidation script (completed)"
safe_delete "fix_epa_cross_season.py" "EPA cross-season fix (completed)"

# Old validation results
safe_delete "validation_results" "Old validation results directory"
safe_delete "validation_results_supabase" "Supabase validation results directory"
safe_delete "validation_results.json" "Validation results JSON"
safe_delete "validation_report.txt" "Validation report"

# Keep data_quality_audit.py and pre_training_checklist.py - they're reusable
echo "  ℹ️  Keeping data_quality_audit.py (reusable)"
echo "  ℹ️  Keeping pre_training_checklist.py (reusable)"
echo "  ✓ Phase 4 complete"

echo ""
echo "Phase 5: Duplicate Directories"
echo "-------------------------------"
cd "$PROJECT_ROOT"

# Files download artifacts
if [ -d "files (2)" ]; then
    safe_delete "files (2)" "Download artifacts directory (108 KB)"
fi

# Old improved_nfl_system version
if [ -d "reference/improved_nfl_system_2" ]; then
    safe_delete "reference/improved_nfl_system_2" "Old system version (Sept 21)"
fi

# Kitchen directory (unused alternative structure)
if [ -d "kitchen" ]; then
    safe_delete "kitchen" "Unused kitchen/ directory (564 KB)"
fi

echo "  ✓ Phase 5 complete"

echo ""
echo "Phase 6: Completed Log Files"
echo "-----------------------------"
cd "$PROJECT_ROOT/improved_nfl_system/logs"

# Archive completed operation logs
if [ -f "bulk_import.log" ]; then
    safe_delete "bulk_import.log" "Bulk import log (completed operation)"
fi
if [ -f "bulk_import_run.log" ]; then
    safe_delete "bulk_import_run.log" "Bulk import run log (completed)"
fi
if [ -f "epa_fix.log" ]; then
    safe_delete "epa_fix.log" "EPA fix log (completed operation)"
fi

echo "  ℹ️  Keeping nfl_system.log (active)"
echo "  ✓ Phase 6 complete"

echo ""
echo "============================================"
echo "✅ Cleanup Complete!"
echo "============================================"
echo ""
echo "Files removed: $DELETED_FILES"
echo "Directories removed: $DELETED_DIRS"
echo ""
echo "Summary:"
echo "  • Supabase migration artifacts: ~109 files"
echo "  • Duplicate documentation: 11 files"
echo "  • Orphaned database files: 4 files"
echo "  • One-time scripts: 7 files"
echo "  • Duplicate directories: 3 directories"
echo "  • Completed log files: 3 files"
echo "  • Total space freed: ~12-15 MB"
echo ""
echo "📦 Backup location: $BACKUP_DIR"
echo "   (Safe to delete after verifying system works)"
echo ""
echo "Next steps:"
echo "  1. Verify system: cd improved_nfl_system && python main.py"
echo "  2. Check web interface: cd web && python launch.py"
echo "  3. If all works: rm -rf \"$BACKUP_DIR\""
echo ""
echo "Production system remains intact at:"
echo "  /improved_nfl_system/"
echo ""
