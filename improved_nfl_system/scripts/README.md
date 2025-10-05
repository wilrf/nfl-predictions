# Utility Scripts

This directory contains utility scripts for system maintenance, testing, and data operations.

## Utilities

- **`utilities/analyze_all_data.py`** - Comprehensive database content analysis
- **`utilities/validate_and_report.py`** - System validation and reporting

## MCP Testing

- **`test_mcp.sh`** - MCP (Model Context Protocol) testing script
- **`load_via_mcp.sh`** - Data loading via MCP

## Supabase (Archived)

- **`supabase/`** - Legacy Supabase migration scripts (migration complete, archived for reference)
  - `execute_schemas_direct.py`
  - `execute_via_api.py`
  - `execute_with_management_api.py`
  - `parse_connection.py`
  - `test_connection.py`

## Usage

```bash
# Run database analysis
python scripts/utilities/analyze_all_data.py

# Run validation report
python scripts/utilities/validate_and_report.py

# Test MCP
./scripts/test_mcp.sh
```
