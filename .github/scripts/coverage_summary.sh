#!/bin/bash
set -e

# Script to generate coverage report summary for GitHub Actions
# Usage: ./coverage_summary.sh

echo "📊 Generating coverage report summary..."

# Define the summary content
SUMMARY_CONTENT="## Coverage Report Summary
Coverage report has been generated and uploaded as artifacts.

📊 **Artifacts Available:**
- HTML Coverage Report: Download the 'coverage-report' artifact
- XML Coverage Data: Download the 'coverage-xml' artifact
- Test Results: Download the 'test-results' artifact

The HTML coverage report provides detailed line-by-line coverage information."

# Output to console for step logs
echo "$SUMMARY_CONTENT"
echo ""
echo "✅ Writing summary to GitHub Job Summary..."

# Output to GitHub Step Summary
echo "$SUMMARY_CONTENT" >> $GITHUB_STEP_SUMMARY

echo "✅ Coverage report summary completed"
