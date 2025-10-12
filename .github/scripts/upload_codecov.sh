#!/bin/bash
set -e

# Alternative: Upload coverage to codecov for automatic badge generation
# Add this to your workflow after running tests

echo "Uploading coverage to Codecov..."

# Upload coverage report (requires CODECOV_TOKEN secret)
if [ -n "$CODECOV_TOKEN" ]; then
  curl -Os https://uploader.codecov.io/latest/linux/codecov
  chmod +x codecov
  ./codecov -t ${CODECOV_TOKEN}
  echo "✅ Coverage uploaded to Codecov"
  echo "📍 Badge URL: https://codecov.io/gh/jbsoftware-io/gen-ai-demo-app/branch/main/graph/badge.svg"
else
  echo "⚠️  CODECOV_TOKEN not set - skipping codecov upload"
  echo "💡 Set CODECOV_TOKEN as a repository secret to enable automatic coverage badges"
fi