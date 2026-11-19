# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Synthetic, in-memory "internal knowledge base" for the Knowledge Assistant MCP
server.

IMPORTANT — this is 100% fake demo content:
  • No real company data, no real PII, no confidential material.
  • Documents are obviously synthetic internal wiki / policy / runbook pages.
  • Content lives in module-level structures and is rebuilt from SEED on process
    start. There is no persistence and nothing is written back.

Your own files are added alongside the seed: everything under DOCS_DIR is loaded
by loader.py (PDF, Word, Markdown, text, HTML) into the same working set. Set
INCLUDE_SAMPLE_DOCS=false to serve only your files and drop the synthetic seed.

This module is the stand-in for an "internal vector database + document store".
In a real deployment you would replace `SEED_DOCUMENTS` with a connector to your
actual document store (Confluence, SharePoint, S3, a managed vector DB, etc.) and
`vectorstore.py` with your real embedding + retrieval backend.

Do NOT treat any of this as authoritative company policy.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

from . import loader

logger = logging.getLogger(__name__)

INCLUDE_SAMPLE_DOCS = os.getenv("INCLUDE_SAMPLE_DOCS", "true").strip().lower() not in ("0", "false", "no")

# ---------------------------------------------------------------------------
# Seed documents — the immutable source of truth. Each document has:
#   doc_id      stable identifier used by get_document()
#   title       human-readable title
#   collection  logical grouping (hr / it / security / product / onboarding)
#   tags        free-form keywords
#   updated     ISO date the doc was last "updated" (synthetic)
#   text        the body; paragraphs are separated by a blank line and are used
#               as retrieval chunks by the vector store.
# ---------------------------------------------------------------------------

_SEED_DOCUMENTS: list[dict[str, Any]] = [
    {
        "doc_id": "HR-001",
        "title": "Paid Time Off (PTO) Policy",
        "collection": "hr",
        "tags": ["pto", "vacation", "leave", "time off", "benefits"],
        "updated": "2026-01-15",
        "text": (
            "Full-time employees accrue paid time off (PTO) at a rate of 1.5 days "
            "per month, for a total of 18 days per calendar year. PTO begins "
            "accruing on the first day of employment and is available for use "
            "immediately.\n\n"
            "Unused PTO of up to 5 days may be carried over into the next calendar "
            "year. Any balance above 5 days on December 31 is forfeited, so "
            "employees are encouraged to plan time off in advance.\n\n"
            "To request PTO, submit a request in the HR portal at least two weeks "
            "before the intended start date. Requests are approved by your direct "
            "manager. For absences longer than 10 consecutive business days, "
            "additional approval from your department lead is required."
        ),
    },
    {
        "doc_id": "HR-002",
        "title": "Remote Work and Hybrid Schedule Guidelines",
        "collection": "hr",
        "tags": ["remote", "hybrid", "work from home", "wfh", "schedule"],
        "updated": "2026-02-03",
        "text": (
            "The company operates on a hybrid model. Employees are expected to be "
            "in the office at least two days per week, with Tuesday and Thursday "
            "designated as company-wide collaboration days.\n\n"
            "Fully remote arrangements are available on a case-by-case basis and "
            "must be approved by both your manager and HR. Remote employees are "
            "responsible for maintaining a secure and reliable internet connection "
            "and a distraction-free workspace during core hours (10:00–16:00 local "
            "time).\n\n"
            "Home-office equipment stipends of up to 500 USD per year are available "
            "to reimburse the purchase of a monitor, keyboard, chair, or similar "
            "equipment. Submit receipts through the expense portal under the "
            "'Home Office' category."
        ),
    },
    {
        "doc_id": "HR-003",
        "title": "Expense Reimbursement Procedure",
        "collection": "hr",
        "tags": ["expenses", "reimbursement", "travel", "receipts", "finance"],
        "updated": "2025-12-10",
        "text": (
            "Business expenses must be submitted through the expense portal within "
            "30 days of being incurred. Each expense over 25 USD requires an "
            "itemized receipt. Expenses submitted more than 60 days late may be "
            "denied.\n\n"
            "Approved categories include travel, lodging, client meals, software "
            "subscriptions, and home-office equipment. Alcohol is only reimbursable "
            "as part of an approved client-entertainment event.\n\n"
            "Reimbursements are paid out with the next payroll cycle after manager "
            "approval. Mileage for personal-vehicle business travel is reimbursed "
            "at the standard company rate of 0.67 USD per mile."
        ),
    },
    {
        "doc_id": "IT-001",
        "title": "VPN Setup and Access Runbook",
        "collection": "it",
        "tags": ["vpn", "remote access", "network", "setup", "runbook"],
        "updated": "2026-03-01",
        "text": (
            "The company VPN provides secure access to internal systems from "
            "outside the corporate network. All remote employees must connect "
            "through the VPN before accessing internal wikis, dashboards, or "
            "source-code repositories.\n\n"
            "To set up the VPN: (1) install the AnyConnect client from the software "
            "center, (2) enter the gateway address vpn.example-corp.internal, "
            "(3) authenticate with your SSO credentials, and (4) approve the "
            "multi-factor push notification on your registered device.\n\n"
            "If the VPN fails to connect, first confirm you are not already on the "
            "office network, then restart the client. Persistent failures with "
            "error 'AnyConnect was not able to establish a connection' usually "
            "indicate an expired certificate — contact the IT service desk to have "
            "your device certificate reissued."
        ),
    },
    {
        "doc_id": "IT-002",
        "title": "Password and Multi-Factor Authentication Policy",
        "collection": "it",
        "tags": ["password", "mfa", "2fa", "authentication", "sso"],
        "updated": "2026-02-20",
        "text": (
            "Passwords for all corporate accounts must be at least 14 characters "
            "and include a mix of upper- and lower-case letters, numbers, and "
            "symbols. Passwords may not be reused across systems and expire every "
            "180 days.\n\n"
            "Multi-factor authentication (MFA) is mandatory for all employees. The "
            "approved second factors are the company authenticator app and hardware "
            "security keys. SMS-based codes are being phased out and should not be "
            "used for new enrollments.\n\n"
            "If you lose access to your MFA device, contact the IT service desk to "
            "begin identity re-verification. Never share one-time codes with anyone, "
            "including IT staff — legitimate support will never ask for them."
        ),
    },
    {
        "doc_id": "IT-003",
        "title": "Requesting Software and Hardware",
        "collection": "it",
        "tags": ["software", "hardware", "procurement", "laptop", "license"],
        "updated": "2026-01-28",
        "text": (
            "Standard software (browsers, office suite, collaboration tools) is "
            "pre-approved and can be installed directly from the software center. "
            "Non-standard or paid software requires a request in the IT portal and "
            "manager approval before a license is purchased.\n\n"
            "Hardware refresh cycles are three years for laptops and four years for "
            "monitors. Early replacements require justification such as a hardware "
            "failure or a role change with new requirements.\n\n"
            "Requests are typically fulfilled within five business days. Urgent "
            "requests can be escalated by flagging the ticket as 'business "
            "critical' and notifying your manager."
        ),
    },
    {
        "doc_id": "SEC-001",
        "title": "Data Classification and Handling Standard",
        "collection": "security",
        "tags": ["data classification", "confidential", "handling", "compliance"],
        "updated": "2026-02-11",
        "text": (
            "Company data is classified into four levels: Public, Internal, "
            "Confidential, and Restricted. The classification determines how the "
            "data may be stored, shared, and transmitted.\n\n"
            "Public data may be shared freely. Internal data is for employees only. "
            "Confidential data (such as unreleased product plans and financials) may "
            "only be shared on a need-to-know basis and must be encrypted in "
            "transit. Restricted data (such as credentials and personal data subject "
            "to regulation) must additionally be encrypted at rest and access-logged.\n\n"
            "Never place Confidential or Restricted data in unmanaged locations such "
            "as personal cloud drives, personal email, or public code repositories. "
            "Report suspected mishandling to the security team within 24 hours."
        ),
    },
    {
        "doc_id": "SEC-002",
        "title": "Security Incident Reporting Runbook",
        "collection": "security",
        "tags": ["incident", "phishing", "breach", "reporting", "runbook"],
        "updated": "2026-03-05",
        "text": (
            "A security incident is any event that may compromise the "
            "confidentiality, integrity, or availability of company systems or data "
            "— including phishing, malware, lost devices, and unauthorized access.\n\n"
            "If you suspect an incident, do not attempt to investigate it yourself. "
            "Immediately report it to the security team via the '#security-incident' "
            "channel or the 24/7 hotline listed in the security portal. Preserve "
            "evidence: do not delete suspicious emails or power off an affected "
            "machine unless instructed.\n\n"
            "For suspected phishing, use the 'Report Phishing' button in the mail "
            "client. The security team triages every report and will follow up if "
            "action is required. Fast reporting dramatically reduces the impact of "
            "an incident."
        ),
    },
    {
        "doc_id": "PROD-001",
        "title": "Acme Analytics Platform — Product Overview",
        "collection": "product",
        "tags": ["product", "analytics", "features", "overview", "acme"],
        "updated": "2026-02-25",
        "text": (
            "The Acme Analytics Platform is our flagship product for real-time "
            "business intelligence. It ingests event data from web, mobile, and "
            "backend sources and provides dashboards, alerting, and ad-hoc query "
            "capabilities.\n\n"
            "Core features include: customizable dashboards, a drag-and-drop report "
            "builder, anomaly-detection alerts, role-based access control, and an "
            "SQL-compatible query interface. Data is retained for 13 months on the "
            "standard plan and 36 months on the enterprise plan.\n\n"
            "The platform integrates with common data warehouses and supports "
            "single sign-on via SAML and OIDC. A public REST API and client SDKs "
            "for Python and JavaScript are available for programmatic access."
        ),
    },
    {
        "doc_id": "PROD-002",
        "title": "Acme Analytics — Service Level and Support Tiers",
        "collection": "product",
        "tags": ["sla", "support", "uptime", "tiers", "enterprise"],
        "updated": "2026-01-30",
        "text": (
            "The Acme Analytics Platform offers three support tiers: Standard, "
            "Business, and Enterprise. The tier determines response times and the "
            "availability commitment.\n\n"
            "Standard includes email support with a next-business-day response and a "
            "99.5% monthly uptime target. Business adds live chat with a four-hour "
            "response during business hours and a 99.9% uptime SLA. Enterprise adds "
            "24/7 phone support, a dedicated technical account manager, a one-hour "
            "response for critical issues, and a 99.95% uptime SLA with service "
            "credits.\n\n"
            "Uptime is measured monthly and excludes scheduled maintenance, which is "
            "announced at least 72 hours in advance and performed in low-traffic "
            "windows."
        ),
    },
    {
        "doc_id": "ONB-001",
        "title": "New Hire First-Week Onboarding Checklist",
        "collection": "onboarding",
        "tags": ["onboarding", "new hire", "checklist", "first week", "getting started"],
        "updated": "2026-02-08",
        "text": (
            "Welcome aboard! Your first week is focused on getting access, meeting "
            "your team, and completing required training. Your manager and an "
            "assigned onboarding buddy will help you through it.\n\n"
            "Day one: collect your laptop from IT, complete SSO and MFA enrollment, "
            "and sign the acceptable-use policy. Day two to three: set up the VPN, "
            "join your team's channels, and complete the security-awareness and "
            "data-handling training modules.\n\n"
            "By the end of week one you should have completed all mandatory "
            "compliance training, met with your manager to review your 30-day goals, "
            "and submitted any home-office equipment requests. If you are blocked on "
            "access, contact the IT service desk early rather than waiting."
        ),
    },
]


# ---------------------------------------------------------------------------
# Working set (rebuilt from the seed + DOCS_DIR files on import / reset). Kept
# read-only in this demo — there are no mutating tools — but structured the same
# way as banking-assistant so a real deployment can swap in a live-loading backend.
# ---------------------------------------------------------------------------

DOCUMENTS: dict[str, dict[str, Any]] = {}


def reset() -> None:
    """Rebuild the working document set from the seed and the docs folder."""
    global DOCUMENTS
    seed = _SEED_DOCUMENTS if INCLUDE_SAMPLE_DOCS else []
    DOCUMENTS = {d["doc_id"]: copy.deepcopy(d) for d in seed}
    for doc in loader.load_documents():
        if doc["doc_id"] in DOCUMENTS:
            logger.warning("Skipping %s: doc_id clashes with a sample document.", doc["doc_id"])
            continue
        DOCUMENTS[doc["doc_id"]] = doc


# Populate on import.
reset()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return a full document by id, or None if it does not exist."""
    doc = DOCUMENTS.get(doc_id)
    return copy.deepcopy(doc) if doc else None


def all_documents() -> list[dict[str, Any]]:
    """Return every document (deep-copied)."""
    return [copy.deepcopy(d) for d in DOCUMENTS.values()]


def documents_in(collection: str | None = None) -> list[dict[str, Any]]:
    """Return documents, optionally filtered to a single collection."""
    docs = DOCUMENTS.values()
    if collection:
        docs = [d for d in docs if d["collection"] == collection]
    return [copy.deepcopy(d) for d in docs]


def collections() -> dict[str, int]:
    """Return a mapping of collection name -> document count."""
    counts: dict[str, int] = {}
    for d in DOCUMENTS.values():
        counts[d["collection"]] = counts.get(d["collection"], 0) + 1
    return counts


def document_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Return just the metadata for a document (no body text)."""
    return {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "collection": doc["collection"],
        "tags": doc["tags"],
        "updated": doc["updated"],
        **({"source": doc["source"]} if "source" in doc else {}),
    }
