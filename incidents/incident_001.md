---
incident_id: INC-2024-001
title: Payment Service Outage
severity: P1
date: 2024-01-15
duration_minutes: 45
services_affected: [payment-api, checkout-service]
root_cause: Database connection pool exhaustion
---

# Incident Summary

On January 15, 2024, the payment service experienced a complete outage
lasting 45 minutes. Users were unable to complete purchases, resulting
in approximately $150,000 in lost revenue.

## Timeline

- **14:32 UTC**: First alerts fired for elevated 5xx errors on payment-api
- **14:35 UTC**: On-call engineer acknowledged alert
- **14:42 UTC**: Identified connection pool exhaustion in database metrics
- **14:55 UTC**: Root cause identified as connection leak in new deployment
- **15:02 UTC**: Rollback initiated
- **15:17 UTC**: Service restored to normal operation

## Root Cause Analysis

The deployment from January 14 introduced a code path that didn't properly
release database connections in error scenarios. Under normal load, this
wasn't apparent, but a traffic spike at 14:30 UTC triggered the error
path repeatedly, exhausting the connection pool within minutes.

## Contributing Factors

1. Integration tests didn't cover the error path with connection handling
2. Connection pool metrics weren't included in deployment validation
3. Canary deployment was too short to catch the slow leak

## Action Items

- [ ] Add integration tests for error path connection handling
- [ ] Include connection pool metrics in deployment dashboards
- [ ] Extend canary period from 10 to 30 minutes for payment service
- [ ] Implement connection pool circuit breaker

## Lessons Learned

Resource leaks are insidious because they don't manifest immediately.
We need better testing of error paths, particularly for resource management.
The team also noted that our rollback process worked well once initiated,
but detection took too long.
