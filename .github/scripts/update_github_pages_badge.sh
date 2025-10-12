#!/bin/bash
set -e

# Script to update GitHub Pages with coverage badge
# This creates a separate commit to gh-pages branch

echo "Updating GitHub Pages with coverage badge..."

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

# Create a temporary directory for gh-pages content
mkdir -p gh-pages-content

# Generate the badge JSON for shields.io endpoint
cat > gh-pages-content/coverage-badge.json << EOF
{
  "schemaVersion": 1,
  "label": "coverage",
  "message": "${COVERAGE_PERCENT}%",
  "color": "${COVERAGE_COLOR}"
}
EOF

# Create a simple index.html for the GitHub Pages site
cat > gh-pages-content/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Gen AI Demo App - Coverage Badge</title>
</head>
<body>
    <h1>Coverage Badge Endpoint</h1>
    <p>Current coverage: ${COVERAGE_PERCENT}%</p>
    <p>Badge JSON: <a href="coverage-badge.json">coverage-badge.json</a></p>
    <p>Last updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)</p>
</body>
</html>
EOF

# If we're in GitHub Actions, deploy to gh-pages
if [ -n "$GITHUB_ACTIONS" ]; then
  echo "🚀 Deploying to GitHub Pages..."
  
  # Debug: Show repository information
  echo "🔍 Debug Info:"
  echo "  - GITHUB_REPOSITORY: ${GITHUB_REPOSITORY:-not set}"
  echo "  - Current remote URL: $(git remote get-url origin)"
  echo "  - Current branch: $(git branch --show-current)"
  
  # Ensure we're using the correct repository
  if [ -n "$GITHUB_REPOSITORY" ]; then
    EXPECTED_REMOTE="https://github.com/${GITHUB_REPOSITORY}"
    CURRENT_REMOTE=$(git remote get-url origin)
    if [[ "$CURRENT_REMOTE" != *"$GITHUB_REPOSITORY"* ]]; then
      echo "⚠️  Warning: Remote URL mismatch!"
      echo "   Expected: $EXPECTED_REMOTE"
      echo "   Current:  $CURRENT_REMOTE"
      echo "🔧 Fixing remote URL..."
      git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
    fi
  fi
  
  # Configure git
  git config --local user.email "action@github.com"
  git config --local user.name "GitHub Action"
  
  # Check if gh-pages branch exists
  if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    echo "📋 Checking out existing gh-pages branch"
    git fetch origin gh-pages
    git checkout -b gh-pages origin/gh-pages
  else
    echo "🆕 Creating new gh-pages branch"
    git checkout --orphan gh-pages
    git rm -rf .
  fi
  
  # Copy our content
  cp gh-pages-content/* .
  
  # Commit and push
  git add .
  git commit -m "Update coverage badge: ${COVERAGE_PERCENT}% [skip ci]" || echo "No changes to commit"
  git push origin gh-pages
  
  echo "✅ GitHub Pages updated with coverage: ${COVERAGE_PERCENT}%"
  
  # Use GitHub environment variables if available, otherwise fallback to hardcoded
  if [ -n "$GITHUB_REPOSITORY" ]; then
    REPO_OWNER=$(echo $GITHUB_REPOSITORY | cut -d'/' -f1)
    REPO_NAME=$(echo $GITHUB_REPOSITORY | cut -d'/' -f2)
    echo "📂 Using repository: ${REPO_OWNER}/${REPO_NAME}"
    BADGE_URL="https://img.shields.io/endpoint?url=https://${REPO_OWNER}.github.io/${REPO_NAME}/coverage-badge.json"
  else
    echo "📂 Using fallback repository: jbsoftware-io/gen-ai-demo-app"
    BADGE_URL="https://img.shields.io/endpoint?url=https://jbsoftware-io.github.io/gen-ai-demo-app/coverage-badge.json"
  fi
  
  echo "📍 Badge URL: ${BADGE_URL}"
else
  echo "📁 Generated GitHub Pages content in gh-pages-content/"
  echo "💡 In GitHub Actions, this will be deployed to gh-pages branch"
fi