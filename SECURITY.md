# Security policy

## Supported versions

Only the latest release and `main` receive security fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting instead:
**Security → Report a vulnerability** on this repository
(https://github.com/greenclaw/gpuwarden/security/advisories/new).

Include the version or commit, what you did, and what happened. You should get a first answer
within a week.

## Scope

GPUWarden holds a RunPod API key and the bearer key of the pods it creates. Reports about key
exposure (logs, process arguments, files), unauthenticated endpoints, and pods left running or
billing against the operator's intent are especially welcome.
