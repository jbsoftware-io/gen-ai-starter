#!/bin/bash
set -e

# Script to generate a coverage badge SVG file
# This doesn't modify the repo, just creates an artifact

echo "Generating coverage badge..."

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

# Create badges directory
mkdir -p badges

# Generate SVG badge using shields.io API
BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_PERCENT}%25-${COVERAGE_COLOR}.svg"
curl -s "$BADGE_URL" > badges/coverage.svg

# Also create a JSON file with the coverage data
cat > badges/coverage.json << EOF
{
  "coverage": ${COVERAGE_PERCENT},
  "color": "${COVERAGE_COLOR}",
  "url": "${BADGE_URL}",
  "generated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "✅ Generated coverage badge: ${COVERAGE_PERCENT}%"
echo "📁 Badge files created in badges/ directory"