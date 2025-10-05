#!/bin/bash
# ============================================================================
# Comprehensive NFL Data Setup Script
# ============================================================================
# Purpose: One-command setup for comprehensive NFL data warehouse
# Usage: ./setup_comprehensive_data.sh [--skip-pbp] [--dry-run]
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
SKIP_PBP=false
DRY_RUN=false
START_YEAR=2016
END_YEAR=2024

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-pbp)
            SKIP_PBP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --start-year=*)
            START_YEAR="${arg#*=}"
            shift
            ;;
        --end-year=*)
            END_YEAR="${arg#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-pbp              Skip play-by-play import (faster, ~5 min)"
            echo "  --dry-run               Show what would be done without doing it"
            echo "  --start-year=YYYY       Start year (default: 2016)"
            echo "  --end-year=YYYY         End year (default: 2024)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Full import (15-30 min)"
            echo "  $0 --skip-pbp           # Fast import without play-by-play (5 min)"
            echo "  $0 --dry-run            # Preview what will happen"
            exit 0
            ;;
    esac
done

# Print header
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}      COMPREHENSIVE NFL DATA WAREHOUSE SETUP${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Date Range: ${GREEN}$START_YEAR-$END_YEAR${NC}"
echo -e "Skip Play-by-Play: ${GREEN}$SKIP_PBP${NC}"
echo -e "Dry Run: ${GREEN}$DRY_RUN${NC}"
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
    echo "Steps that would be executed:"
    echo "  1. Create backups of current database and ML data"
    echo "  2. Run schema migration (database/nfl_comprehensive.db)"
    echo "  3. Import schedules (~2,400 games)"
    if [ "$SKIP_PBP" = false ]; then
        echo "  4. Import play-by-play (~432,000 plays) - 15 min"
    else
        echo "  4. [SKIPPED] Play-by-play import"
    fi
    echo "  5. Import NGS data (~25,000 records)"
    echo "  6. Import injuries (~54,000 records)"
    echo "  7. Import snap counts (~234,000 records)"
    echo "  8. Import rosters (~363,000 records)"
    echo "  9. Import depth charts (~335,000 records)"
    echo "  10. Import officials (~17,000 records)"
    echo "  11. Generate import summary"
    echo ""
    echo "Estimated total time: $([ "$SKIP_PBP" = true ] && echo "5 minutes" || echo "15-30 minutes")"
    echo "Estimated database size: $([ "$SKIP_PBP" = true ] && echo "~150 MB" || echo "~500 MB")"
    exit 0
fi

# Confirm with user
echo -e "${YELLOW}⚠️  This will:${NC}"
echo "  - Create a new comprehensive database (~500 MB)"
echo "  - Import ~1.13 million records from nfl_data_py"
echo "  - Take approximately $([ "$SKIP_PBP" = true ] && echo "5 minutes" || echo "15-30 minutes")"
echo ""
read -p "Continue? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${RED}Aborted by user${NC}"
    exit 1
fi

# Step 1: Create logs directory
echo -e "\n${BLUE}[Step 1/4] Creating logs directory...${NC}"
mkdir -p logs
mkdir -p database/backups

# Step 2: Backup current system
echo -e "\n${BLUE}[Step 2/4] Backing up current system...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -f "database/nfl_betting.db" ]; then
    echo "  Backing up database..."
    cp database/nfl_betting.db database/backups/nfl_betting_backup_$TIMESTAMP.db
    echo -e "  ${GREEN}✅ Database backup created${NC}"
else
    echo -e "  ${YELLOW}⚠️  No existing database found (this is fine for first run)${NC}"
fi

if [ -d "ml_training_data" ]; then
    echo "  Backing up ML training data..."
    tar -czf database/backups/ml_training_data_backup_$TIMESTAMP.tar.gz ml_training_data/ 2>/dev/null || true
    echo -e "  ${GREEN}✅ ML data backup created${NC}"
fi

# Step 3: Run migration
echo -e "\n${BLUE}[Step 3/4] Running schema migration...${NC}"
python3 migrate_to_comprehensive_schema.py \
    --old-db database/nfl_betting.db \
    --new-db database/nfl_comprehensive.db

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Migration complete!${NC}"
else
    echo -e "${RED}❌ Migration failed!${NC}"
    echo "Check logs/bulk_import_comprehensive.log for details"
    exit 1
fi

# Step 4: Run bulk import
echo -e "\n${BLUE}[Step 4/4] Importing NFL data...${NC}"
echo -e "${YELLOW}⚠️  This may take a while. Get some coffee! ☕${NC}"
echo ""

IMPORT_CMD="python3 bulk_import_all_data.py --db database/nfl_comprehensive.db --start-year $START_YEAR --end-year $END_YEAR"

if [ "$SKIP_PBP" = true ]; then
    IMPORT_CMD="$IMPORT_CMD --skip-pbp"
fi

eval $IMPORT_CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Import complete!${NC}"
else
    echo -e "\n${RED}❌ Import failed!${NC}"
    echo "Check logs/bulk_import_comprehensive.log for details"
    exit 1
fi

# Success summary
echo -e "\n${BLUE}============================================================================${NC}"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Database: database/nfl_comprehensive.db"
echo "Backups: database/backups/"
echo "Logs: logs/bulk_import_comprehensive.log"
echo "Summary: logs/bulk_import_summary.json"
echo ""

# Show database info
DB_SIZE=$(du -h database/nfl_comprehensive.db | cut -f1)
echo -e "Database size: ${GREEN}$DB_SIZE${NC}"
echo ""

# Show summary if exists
if [ -f "logs/bulk_import_summary.json" ]; then
    echo "Import Summary:"
    python3 -c "
import json
with open('logs/bulk_import_summary.json', 'r') as f:
    summary = json.load(f)
    print(f\"  Total records: {summary['total_records']:,}\")
    print(f\"  Date range: {summary['date_range']}\")
    print(f\"  Database size: {summary['database_size_mb']} MB\")
"
    echo ""
fi

echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Validate data: sqlite3 database/nfl_comprehensive.db 'SELECT COUNT(*) FROM fact_games;'"
echo "  2. Create feature aggregations (agg_team_epa_stats, etc.)"
echo "  3. Generate ML training data with 50+ features"
echo "  4. Train models!"
echo ""
echo -e "${GREEN}Happy modeling! 🚀${NC}"
