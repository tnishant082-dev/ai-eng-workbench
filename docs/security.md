# Security

## Implemented (local)

- API key via `X-API-Key` (dev key in `.env.example`)
- Simple per-IP rate limit on `/api/*`
- Secrets only via environment — never committed
- Prompt-injection notes in corpus (`22_prompt_injection.md`)
- Calculator tool uses AST whitelist (no `eval`)

## Not implemented (on purpose)

- OAuth/OIDC, RBAC, row-level tenancy
- Network egress controls / VPC
- Prompt firewall / output filtering beyond heuristics
- Encrypted-at-rest SQLite

Treat this as a **portfolio sandbox**, not a hardened multi-tenant SaaS.
