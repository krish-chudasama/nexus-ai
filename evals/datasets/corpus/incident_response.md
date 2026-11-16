# Incident Response Runbook

## Severity levels
SEV1 means a full outage or data loss affecting most customers; the on-call engineer must acknowledge within 15 minutes. SEV2 means a major feature is degraded; acknowledgement is required within 1 hour. SEV3 covers minor issues and is handled during business hours.

## On-call rotation
The on-call rotation changes every week on Monday at 09:00 UTC. The primary on-call engineer is backed up by a secondary engineer who is paged after 10 minutes without acknowledgement.

## Communication
All incidents are coordinated in the #incidents Slack channel. The incident commander posts a status update every 30 minutes for SEV1 incidents. Customer-facing updates are published on the status page by the communications lead.

## Postmortems
A blameless postmortem must be published within 5 business days after any SEV1 or SEV2 incident. It must include a timeline, root cause, impact, and at least one corrective action with an owner.
