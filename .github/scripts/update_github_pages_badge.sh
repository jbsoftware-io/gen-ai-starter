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
  # Force the repository to be gen-ai-starter regardless of what GitHub Actions thinks
  CORRECT_REPO="jbsoftware-io/gen-ai-starter"
  EXPECTED_REMOTE="https://github.com/${CORRECT_REPO}"
  CURRENT_REMOTE=$(git remote get-url origin)
  
  echo "🔍 Repository verification:"
  echo "  - GitHub Actions thinks: ${GITHUB_REPOSITORY:-not set}"
  echo "  - We want: $CORRECT_REPO"
  echo "  - Current remote: $CURRENT_REMOTE"
  echo "  - Target remote: $EXPECTED_REMOTE"
  
  if [[ "$CURRENT_REMOTE" != *"gen-ai-starter"* ]]; then
    echo "🔧 Fixing remote URL to point to correct repository..."
    git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${CORRECT_REPO}.git"
    echo "✅ Remote URL updated to: $(git remote get-url origin)"
  else
    echo "✅ Remote URL is already correct"
  fi
  
  # Configure git
  git config --local user.email "action@github.com"
  git config --local user.name "GitHub Action"
  
  # Check if gh-pages branch exists
  if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
    echo "📋 Checking out existing gh-pages branch"
    git fetch origin gh-pages || echo "Warning: Could not fetch gh-pages"
    git checkout -b gh-pages origin/gh-pages || {
      echo "⚠️ Could not checkout existing gh-pages, creating new one"
      git checkout --orphan gh-pages
      git rm -rf . || echo "No files to remove"
    }
  else
    echo "🆕 Creating new gh-pages branch"
    git checkout --orphan gh-pages
    git rm -rf . || echo "No files to remove"
  fi
  
  # Copy our content
  cp gh-pages-content/* .
  
  # Commit and push
  git add .
  git commit -m "Update coverage badge: ${COVERAGE_PERCENT}% [skip ci]" || echo "No changes to commit"
  
  # Force push to handle any conflicts from repository rename
  echo "🚀 Pushing to gh-pages branch..."
  git push origin gh-pages --force || {
    echo "⚠️ Force push failed, trying regular push..."
    git push origin gh-pages || {
      echo "❌ Push failed. This might be due to repository rename issues."
      echo "Manual intervention may be required."
      exit 1
    }
  }
  
  echo "✅ GitHub Pages updated with coverage: ${COVERAGE_PERCENT}%"
  
  # Always use the correct repository name for the badge URL
  CORRECT_REPO="jbsoftware-io/gen-ai-starter"
  REPO_OWNER=$(echo $CORRECT_REPO | cut -d'/' -f1)
  REPO_NAME=$(echo $CORRECT_REPO | cut -d'/' -f2)
  echo "📂 Using repository: ${REPO_OWNER}/${REPO_NAME}"
  BADGE_URL="https://img.shields.io/endpoint?url=https://${REPO_OWNER}.github.io/${REPO_NAME}/coverage-badge.json"
  
  echo "📍 Badge URL: ${BADGE_URL}"
else
  echo "📁 Generated GitHub Pages content in gh-pages-content/"
  echo "💡 In GitHub Actions, this will be deployed to gh-pages branch"
fi