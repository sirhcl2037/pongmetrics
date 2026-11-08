# Build identification

Preserve the visible commit-based build identifier and UTC build timestamp in the webpage footer. The user relies on them to distinguish versions deployed on Microsoft Azure. Generate the identifier automatically at build time for every commit; do not maintain it manually or remove it during UI changes.

Vite reads Git metadata when available, or `PONGMETRICS_COMMIT_SHA`, Azure Pipelines `BUILD_SOURCEVERSION`, GitHub Actions `GITHUB_SHA`, or `SCM_COMMIT_ID`. If deployment builds without Git metadata, pass the source commit through one of those environment variables at build time. Uncommitted local builds display `-local`. Deploy the rebuilt frontend assets after each source change. Restart the development server after committing to refresh its build identifier.
