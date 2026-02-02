---
incident_id: INC-2024-003
title: Search Index Corruption
severity: P2
date: 2024-02-20
duration_minutes: 180
services_affected: [search-api, product-catalog, recommendations]
root_cause: Concurrent index rebuild race condition
---

# Incident Summary

On February 20, 2024, the product search functionality returned
incomplete or incorrect results for 3 hours. Users searching for
products saw missing items or items from wrong categories.

## Timeline

- **06:00 UTC**: Nightly index rebuild job started
- **06:15 UTC**: Ad-hoc reindex triggered by catalog team for new category
- **06:20 UTC**: First customer reports of search issues
- **07:00 UTC**: Engineering escalation
- **08:15 UTC**: Race condition identified between rebuild jobs
- **08:45 UTC**: Corrupted index segments identified
- **09:00 UTC**: Full reindex initiated from clean state
- **09:00 UTC**: Search functionality restored

## Root Cause Analysis

Two index rebuild processes ran concurrently, which shouldn't be possible
but was due to a lock file stored on a node that was replaced during
autoscaling. The concurrent writes corrupted index segments, causing
partial matches and incorrect category associations.

## Contributing Factors

1. Distributed lock mechanism relied on local filesystem
2. No validation of index integrity post-rebuild
3. Catalog team unaware of nightly rebuild schedule
4. Alerting only detected complete failures, not data quality issues

## Action Items

- [ ] Migrate index locks to Redis-based distributed locking
- [ ] Implement post-rebuild integrity checks with sample queries
- [ ] Document index rebuild schedule and add to shared calendar
- [ ] Add search result quality metrics to monitoring

## Lessons Learned

Distributed systems need distributed coordination primitives. Local
filesystem locks don't work when nodes can be replaced. We also need
better communication between teams about scheduled jobs that can
conflict with ad-hoc operations.
