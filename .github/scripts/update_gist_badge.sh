#!/bin/bash
set -e

# Script to update a GitHub gist with coverage badge data
# This works with protected branches since it doesn't modify the repo

echo "Updating coverage badge gist..."

# Get coverage percentage
COVERAGE_PERCENT=$(docker-compose run --rm app python .github/scripts/extract_coverage.py | tail -1)

# Determine coverage color based on percentage
if [ "$COVERAGE_PERCENT" -ge 80 ]; then
  COVERAGE_COLOR="brightgreen"
elif [ "$COVERAGE_PERCENT" -ge 60 ]; then
  COVERAGE_COLOR="yellow"
else
  COVERAGE_COLOR="red"
fi

echo "📊 Coverage: ${COVERAGE_PERCENT}% (${COVERAGE_COLOR})"

# Create JSON for shields.io endpoint
cat > coverage-badge.json << EOF
{
  "schemaVersion": 1,
  "label": "coverage",
  "message": "${COVERAGE_PERCENT}%",
  "color": "${COVERAGE_COLOR}"
}
EOF

# If running in GitHub Actions with a gist ID configured
if [ -n "$GITHUB_TOKEN" ] && [ -n "$COVERAGE_GIST_ID" ]; then
  echo "🔄 Updating gist $COVERAGE_GIST_ID..."
  
  curl -s -X PATCH \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/gists/$COVERAGE_GIST_ID" \
    -d '{
      "files": {
        "coverage-badge.json": {
          "content": "'"$(cat coverage-badge.json | sed 's/"/\\"/g' | tr -d '\n')"'"
        }
      }
    }' > /dev/null
  
  echo "✅ Updated gist with coverage: ${COVERAGE_PERCENT}%"
  echo "📍 Badge URL: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/$(echo $COVERAGE_GIST_ID | cut -d'/' -f1)/$COVERAGE_GIST_ID/raw/coverage-badge.json"
else
  echo "⚠️  GITHUB_TOKEN or COVERAGE_GIST_ID not set - skipping gist update"
  echo "💡 To enable automatic badge updates:"
  echo "   1. Create a gist with a file named 'coverage-badge.json'"
  echo "   2. Set COVERAGE_GIST_ID as a repository variable to the gist ID"
  echo "   3. Ensure GITHUB_TOKEN has gist write permissions"
  echo ""
  echo "Coverage badge JSON would be:"
  cat coverage-badge.json
fi