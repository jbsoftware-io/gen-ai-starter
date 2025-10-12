#!/bin/bash
set -e

# Script to update README.md with latest coverage badge
# Usage: ./update_readme_badges.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Updating README with latest badges..."

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

# Create the new coverage badge URL
COVERAGE_BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_PERCENT}%25-${COVERAGE_COLOR}"

echo "📊 New coverage badge URL: $COVERAGE_BADGE_URL"

# Update README.md with the new coverage percentage
sed -i "s|https://img.shields.io/badge/coverage-[0-9]*%25-[a-z]*|${COVERAGE_BADGE_URL}|g" README.md

# Check if we made changes
if git diff --quiet README.md; then
  echo "ℹ️  No changes needed to README.md"
  exit 0
else
  echo "✅ Updated README.md with coverage: ${COVERAGE_PERCENT}%"
  
  # Configure git
  git config --local user.email "action@github.com"
  git config --local user.name "GitHub Action"
  
  # Commit and push changes
  git add README.md
  git commit -m "Update coverage badge to ${COVERAGE_PERCENT}% [skip ci]"
  git push
  
  echo "🚀 Changes pushed to repository"
fi
