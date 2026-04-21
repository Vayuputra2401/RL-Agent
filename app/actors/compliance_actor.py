"""
ComplianceActor — Simulated Compliance Officer for AP Commander.

Responds to HOLD actions with regulatory review verdicts based on:
  - regulatory_framework: SOX | GDPR | INTERNAL_POLICY
  - severity: how serious the identified issue is

Theme #1 (Halluminate — Multi-Actor Environments)
"""

from __future__ import annotations
import random as _random
from typing import Optional

from ..models import ActorResponse, ActorPersona, ActorType


_REGULATIONS = {
    "SOX": {
        "name": "Sarbanes-Oxley Act (SOX) Section 404",
        "cleared": (
            "[COMPLIANCE] SOX review complete. No material weaknesses identified. "
            "The three-way match documentation is adequate for audit purposes. "
            "Invoice may proceed through standard AP process."
        ),
        "flagged": (
            "[COMPLIANCE] SOX Section 404 concern raised. "
            "Inadequate documentation for this transaction creates audit risk. "
            "Invoice must be REJECTED until supporting documentation is complete "
            "and approved by the Controller."
        ),
    },
    "GDPR": {
        "name": "GDPR / Data Handling Policy",
        "cleared": (
            "[COMPLIANCE] GDPR data-handling review passed. "
            "Vendor is on approved processor list. Invoice data handling is compliant. "
            "No restrictions on payment processing."
        ),
        "flagged": (
            "[COMPLIANCE] GDPR concern: vendor not listed in our approved data-processor registry. "
            "Payment must be HELD until vendor DPA (Data Processing Agreement) is signed "
            "and added to the approved list."
        ),
    },
    "INTERNAL_POLICY": {
        "name": "ACME Internal Compliance Policy v4.2",
        "cleared": (
            "[COMPLIANCE] Internal policy review complete. "
            "Invoice complies with all ACME procurement and payment policies. "
            "No compliance blocks identified. Proceed with standard AP workflow."
        ),
        "flagged": (
            "[COMPLIANCE] Internal policy violation detected. "
            "This transaction requires dual approval per Section 7.3 (transactions > $10,000 "
            "with non-preferred vendors). HOLD until secondary approver sign-off is obtained."
        ),
    },
}


class ComplianceActor:
    """
    Simulated Compliance Officer that responds to AP Clerk HOLD actions.

    Usage:
        actor = ComplianceActor(framework="SOX", seed=42)
        response = actor.respond(context)
    """

    def __init__(
        self,
        framework: str = "INTERNAL_POLICY",
        seed: Optional[int] = None,
    ) -> None:
        self.framework = framework if framework in _REGULATIONS else "INTERNAL_POLICY"
        self._rng = _random.Random(seed)

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "ComplianceActor":
        rng = _random.Random(seed)
        return cls(framework=rng.choice(list(_REGULATIONS.keys())), seed=seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def respond(self, context: dict) -> ActorResponse:
        """
        Generate a compliance review response.

        context keys used:
          invoice_total      (float)
          vendor_name        (str)
          has_violation      (bool)  — whether there is a genuine compliance issue
          violation_type     (str)
        """
        has_violation = context.get("has_violation", False)
        inv_total     = context.get("invoice_total", 0.0)
        vendor        = context.get("vendor_name", "Unknown Vendor")
        reg           = _REGULATIONS[self.framework]

        if has_violation:
            msg     = reg["flagged"]
            verdict = "flagged"
        else:
            msg     = reg["cleared"]
            verdict = "cleared"

        # Append transaction specifics
        detail = (
            f" | Transaction: ${inv_total:,.2f} from {vendor} "
            f"| Framework: {reg['name']}"
        )
        full_msg = msg + detail

        return ActorResponse(
            actor_type=ActorType.compliance,
            persona=ActorPersona.honest,
            message=full_msg,
            verdict=verdict,
            context_tag=full_msg,
        )
