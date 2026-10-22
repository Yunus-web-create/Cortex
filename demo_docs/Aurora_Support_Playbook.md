# Aurora Labs — Customer Support Playbook

## Purpose and scope

This playbook defines how the Support team handles customer conversations for Aurora Flow. It covers severity definitions, response targets, escalation, refunds and the most common troubleshooting paths. The playbook is owned by the Head of Support and reviewed quarterly.

## Support channels and hours

- **In-app chat and email** — 07:00 to 19:00 SAST, Monday to Friday.
- **Emergency line** — 24/7 for severity-1 incidents only, reachable from the status page.
- **Community forum** — monitored during business hours.

First response targets are measured in **business hours**. Weekends and South African public holidays do not count.

## Severity definitions

| Severity | Definition | First response | Update cadence |
| --- | --- | --- | --- |
| S1 — Critical | Platform unavailable or data loss for multiple customers | 15 minutes, 24/7 | Every 60 minutes |
| S2 — High | Major feature broken with no workaround; single customer fully blocked | 1 hour | Every 4 business hours |
| S3 — Normal | Feature impaired but a workaround exists | 8 business hours | Every 2 business days |
| S4 — Low | Question, cosmetic issue, feature request | 2 business days | As needed |

Any ticket can be upgraded to S1 by an on-call engineer if the impact justifies it. Downgrades require the Support Lead's approval.

## Response and resolution targets

- Starter plan: standard targets above.
- Growth plan: **S2 first response within 30 minutes**, S3 within 4 business hours.
- Scale and Enterprise: **named support engineer**, S2 within 15 minutes, plus a monthly review call.

Resolution (not just response) targets: S1 — 4 hours; S2 — 2 business days; S3 — 10 business days. These are goals, not contractual SLAs, except where the Enterprise agreement says otherwise.

## The escalation matrix

1. **Tier 1 (Support Specialist)** — triage, known-issue fixes, account questions.
2. **Tier 2 (Support Engineer)** — log analysis, configuration problems, API issues.
3. **Tier 3 (On-call Engineering)** — suspected platform bugs, data integrity, security.

Tier 2 must be engaged within 30 minutes for any S1 or S2. Tier 3 is paged through PagerDuty by the on-call engineer; support agents do not page Tier 3 directly. The **Incident Commander** for any S1 is the on-call engineering lead, not the support agent.

Status updates during S1 go to the customer every **60 minutes**, even when there is no new information.

## Refunds, credits and cancellation

Self-serve plans (Starter, Growth, Scale) can be cancelled at any time from billing settings and remain active until the end of the paid period. The company does **not** provide cash refunds for partial months. Instead, the standard remedy is:

- **14-day money-back guarantee** for first-time subscribers, measured from the first successful payment.
- Pro-rata service credits for downtime, applied against the next invoice.
- Duplicate or accidental charges are refunded in full when reported within **60 days**.

Enterprise agreements follow the signed contract; the standard Enterprise remedy is a service credit, not a cash refund.

Annual plans cancelled mid-term receive a credit for unused full months, minus any discount already applied. Credits expire after 12 months.

## Identity verification before account changes

Support must verify the requester before changing billing details, SSO settings or workspace ownership. Acceptable verification: a signed email from the workspace owner's domain, or a video call plus government ID. **Never** reset MFA on the basis of an email request alone — escalate to the Security team.

## Common troubleshooting paths

### Runs failing with timeout

Check the flow's plan limit: Starter flows stop after 5 minutes, Growth after 15, Scale after 60. Split long work with the *Wait* step, or reduce batch size in loop steps. If a run exceeds the limit on Enterprise, open an S3 ticket and attach the run ID.

### Webhooks not arriving

Confirm the endpoint shows *Active*, that the sender is using the flow's unique URL, and that the request is JSON. Aurora Flow retries failed deliveries 3 times with exponential backoff over 10 minutes. Payload replay is available for 7 days.

### Approval stuck in pending

Approvals escalate automatically after 48 hours. If the approver has left the company, a workspace admin can reassign the gate. If escalation did not fire, capture the run ID and raise an S3 — this was a known defect fixed in v4.1.

### SSO login loop

Usually caused by a clock skew above 5 minutes or an expired SAML certificate. Ask the customer's IT admin to re-upload the certificate in workspace settings. SCIM provisioning issues are Tier 2.

### Rate limit errors from the API

The public API allows 600 requests per minute per account. Clients should honour the `Retry-After` header. Repeated bursts above the limit can trigger a temporary 15-minute block, which Tier 2 can lift.

## Known issues (as of v4.2)

- Mobile push notifications may arrive twice on Android 14 when battery optimisation is disabled. Fix scheduled for v4.3.
- CSV exports over 200,000 rows time out in the browser. Use the API export endpoint instead.
- Workspace names with emoji are rejected by the billing portal but accepted in the app.

## Customer communication standards

- Always acknowledge impact before explaining cause.
- Never speculate about root cause in an S1 update; say what is known and when the next update will come.
- Avoid promising a fix date unless Engineering has committed to one in writing.
- Close tickets with a one-paragraph summary of what was done and links to relevant documentation.

## Feature requests

Feature requests are logged in the product board with the customer's plan tier and estimated deal value. Requests from Scale and Enterprise customers are reviewed weekly by Product. There is no commitment to delivery timelines.
