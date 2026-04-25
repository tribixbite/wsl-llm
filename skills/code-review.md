---
name: Code Review
description: Structured code review checklist and feedback format
---

# Code Review

## Review Checklist

### Correctness
- [ ] Logic handles edge cases (null, empty, boundary values)
- [ ] Error paths return appropriate status codes / messages
- [ ] Async operations have proper error handling
- [ ] No race conditions in concurrent code

### Security
- [ ] User input is validated and sanitized
- [ ] No SQL injection (parameterized queries used)
- [ ] No XSS (output escaped, CSP headers set)
- [ ] Secrets not hardcoded or logged
- [ ] Auth checks on all protected routes

### Performance
- [ ] No N+1 queries or unnecessary DB calls
- [ ] Large lists paginated
- [ ] No blocking operations on hot paths
- [ ] Appropriate caching where beneficial

### Maintainability
- [ ] Functions do one thing
- [ ] Names are descriptive and consistent
- [ ] No dead code or commented-out blocks
- [ ] Types are specific (no `any` in TypeScript)

### Testing
- [ ] Happy path covered
- [ ] Error cases covered
- [ ] Edge cases covered (empty input, large input, unicode)

## Feedback Format

Structure feedback as:

```
## Summary
One-sentence assessment of the change.

## Issues
### 🔴 Must Fix
- **file:line** — Description of critical issue and suggested fix

### 🟡 Should Fix
- **file:line** — Description of important issue

### 💡 Suggestion
- **file:line** — Optional improvement idea

## What's Good
- Highlight well-written code patterns worth keeping
```

## Severity Guide
- **🔴 Must Fix**: Bugs, security issues, data loss risk, broken functionality
- **🟡 Should Fix**: Performance problems, poor error handling, missing validation
- **💡 Suggestion**: Style improvements, refactoring opportunities, better patterns
