# Aurora Flow — Product Guide (v4.2)

## What Aurora Flow is

Aurora Flow is a workflow automation platform for mid-market operations teams. It connects business systems, adds human approval steps, and gives teams a single audit trail. Version 4.2 shipped **12 March 2026** and is the current generally available release.

## Core concepts

**Workspaces** group related flows, members and data. Each customer account starts with one workspace; Enterprise plans support up to 50.

**Flows** are the automation units. A flow starts with a trigger, then runs a sequence of steps: actions (HTTP requests, database writes, notifications) and gates (human approvals, condition checks). A single flow can hold up to **250 steps**.

**Runs** are executions of a flow. The default maximum run duration is **15 minutes**; the hard limit for any run is 2 hours. Long-running work should be split using the *Wait* step, which can pause a flow for up to 30 days.

**Credentials** are encrypted at rest with AES-256 and are never exposed to flow authors after saving. Keys can be scoped per workspace.

## Triggers

- **Webhook** — unique HTTPS endpoint per flow, 10 requests per second per flow.
- **Schedule** — cron-style, minimum interval 1 minute, time zone aware.
- **Event** — native listeners for HubSpot, Salesforce, Xero, Slack, Jira and Google Workspace.
- **Manual** — run on demand from the UI or API.

Webhook payloads are retained for **7 days** on all plans (30 days on Enterprise) for replay and debugging.

## Plans and limits (2026 pricing)

| Plan | Monthly price | Included runs | Max flows | Seats included | Run timeout |
| --- | --- | --- | --- | --- | --- |
| Starter | $49 | 5,000 | 20 | 3 | 5 min |
| Growth | $199 | 40,000 | 100 | 10 | 15 min |
| Scale | $649 | 250,000 | Unlimited | 25 | 60 min |
| Enterprise | Custom | Custom | Unlimited | Custom | 120 min |

Overage is billed at **$0.004 per run** on Starter and Growth, and $0.0025 on Scale. Extra seats are $15 each per month on all self-serve plans. Annual billing saves 18%.

Sandbox environments are available on Scale and Enterprise and are billed at 30% of the production plan price.

## Approvals

Human approval gates can require a specific role, any member of a team, or a named approver. The default approval timeout is **48 hours**, after which the gate escalates to the assignee's manager. Approvers can act from email, Slack or the mobile app. Every decision records the approver, timestamp and IP address.

Approval steps do not consume runs while waiting.

## Integrations

Aurora Flow ships 84 native connectors. The most used are Slack, HubSpot, Salesforce, Xero, Jira, Notion, Google Sheets, Microsoft Teams, Zendesk and Shopify. Anything without a native connector can be reached with the **HTTP step**, which supports OAuth 2.0, API keys, mTLS and custom headers.

The public REST API is rate-limited to **600 requests per minute per account**. API keys are created per workspace and shown only once.

## AI features (beta)

The **Flow Copilot** lets users describe a workflow in natural language and generates a draft flow. Generated flows are always created in draft state and must be reviewed before activation. Copilot is available on Scale and Enterprise plans, limited to 500 generations per month per account; extra generations cost $0.02 each.

The **Anomaly Digest** uses run history to flag flows whose failure rate or duration deviates more than 3 standard deviations from their 30-day baseline. It emails a weekly digest on Mondays at 08:00 in the workspace's time zone.

## Reliability and SLA

Production targets: **99.9% monthly uptime** for the Flow Engine on Growth and above; 99.95% on Enterprise. Scheduled maintenance is announced at least 72 hours in advance and does not count against uptime. Service credits: 10% of monthly fees for uptime between 99.0% and 99.9%, 25% below 99.0%.

Status is published at status.auroraflow.example. Post-incident reviews are published within 5 business days for any severity-1 incident.

## Data, security and compliance

Runs and payloads are retained for **30 days** by default (90 days on Scale, 12 months on Enterprise). Customer data is hosted in the EU (Frankfurt) or the US (Virginia), chosen at workspace creation and not changeable afterwards.

Aurora Flow is SOC 2 Type II certified, GDPR compliant, and offers a signed DPA on request. Data is encrypted in transit with TLS 1.2+ and at rest with AES-256. Backups run every 6 hours and are retained for 35 days.

Single sign-on via SAML 2.0 and SCIM provisioning are included on Scale and Enterprise. Audit logs are retained for 12 months on Enterprise, 90 days on Scale.

## Mobile and accessibility

The mobile app supports approvals, run monitoring and push alerts. It requires iOS 16+ or Android 12+. The web app targets WCAG 2.2 AA; the flow canvas has full keyboard navigation as of v4.1.

## Getting started

Typical time to first production flow is **under one hour**. The onboarding wizard imports a template, connects one system and runs a test execution. Templates are available for employee onboarding, invoice approval, lead routing and churn alerts.
