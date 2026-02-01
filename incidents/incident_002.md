---
incident_id: INC-2024-002
title: Authentication Service Degradation
severity: P2
date: 2024-02-03
duration_minutes: 120
services_affected: [auth-service, user-api, session-manager]
root_cause: Redis cluster failover during maintenance
---

# Incident Summary

On February 3, 2024, authentication services experienced degraded
performance for 2 hours. Login times increased from 200ms to 8 seconds,
causing significant user frustration and support ticket volume.

## Timeline

- **09:00 UTC**: Scheduled Redis maintenance began
- **09:15 UTC**: Primary Redis node taken offline for patching
- **09:16 UTC**: Failover initiated to replica
- **09:18 UTC**: Latency alerts fired for auth-service
- **09:45 UTC**: Identified Redis cluster in degraded state
- **10:30 UTC**: Manual intervention to stabilize cluster
- **11:15 UTC**: Service performance restored

## Root Cause Analysis

The Redis cluster failover didn't complete cleanly. The replica promoted
to primary was under-provisioned compared to the original primary, and
the maintenance runbook didn't account for this. Additionally, the
auth-service connection pool didn't handle the failover gracefully,
requiring connections to timeout before reconnecting to the new primary.

## Contributing Factors

1. Redis replica sizing wasn't validated against primary
2. Maintenance runbook was outdated (last updated 8 months prior)
3. Auth-service lacks Redis connection resilience patterns
4. No load testing of failover scenarios

## Action Items

- [ ] Audit Redis cluster sizing across all replicas
- [ ] Update maintenance runbooks quarterly
- [ ] Implement Redis Sentinel-aware connection handling in auth-service
- [ ] Add failover scenario to quarterly game days

## Lessons Learned

Maintenance windows require the same rigor as deployments. Our runbook
hadn't kept pace with infrastructure changes. We're implementing a
runbook review process tied to infrastructure change management.
