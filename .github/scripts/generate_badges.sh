#!/bin/bash
set -e

# Script to generate coverage and test badges for GitHub Actions
# Usage: ./generate_badges.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Generating badges..."

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

# Create badges directory
mkdir -p badges

# Generate coverage badge URL
COVERAGE_BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_PERCENT}%25-${COVERAGE_COLOR}"
echo "$COVERAGE_BADGE_URL" > badges/coverage-badge-url.txt

# Generate test status badge (assuming tests passed if we got here)
TEST_BADGE_URL="https://img.shields.io/badge/tests-passing-brightgreen"
echo "$TEST_BADGE_URL" > badges/test-badge-url.txt

echo "✅ Generated badges with coverage: ${COVERAGE_PERCENT}%"
echo "📄 Coverage badge: $COVERAGE_BADGE_URL"
echo "📄 Test badge: $TEST_BADGE_URL"
