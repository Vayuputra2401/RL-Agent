"""
ManagerActor — Simulated Finance Manager for AP Commander.

Responds to ESCALATE actions with approval/denial decisions based on:
  - budget_authority: max amount the manager can approve unilaterally
  - risk_appetite:    how strictly they interpret policy
  - pre_approval:     whether manager has already pre-approved this episode's freight

Theme #1 (Halluminate — Multi-Actor Environments)
"""

from __future__ import annotations
import random as _random
from typing import Optional

from ..models import ActorResponse, ActorPersona, ActorType


class ManagerActor:
    """
    Simulated Finance Manager that responds to AP Clerk ESCALATE actions.

    Usage:
        actor = ManagerActor(budget_authority=50000, risk_appetite="moderate", seed=42)
        response = actor.respond(context)
    """

    BUDGET_TIERS = [5_000, 25_000, 50_000, 150_000]

    def __init__(
        self,
        budget_authority: float = 50_000,
        risk_appetite: str = "moderate",
        pre_approved: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.budget_authority = budget_authority
        self.risk_appetite    = risk_appetite   # "conservative" | "moderate" | "aggressive"
        self.pre_approved     = pre_approved
        self._rng = _random.Random(seed)

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "ManagerActor":
        rng = _random.Random(seed)
        return cls(
            budget_authority=rng.choice(cls.BUDGET_TIERS),
            risk_appetite=rng.choice(["conservative", "moderate", "aggressive"]),
            pre_approved=rng.random() < 0.5,
            seed=seed,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def respond(self, context: dict) -> ActorResponse:
        """
        Generate a manager response based on episode context.

        context keys used:
          invoice_total   (float)
          freight_charge  (float)
          freight_cap     (float)
          violation_type  (str: "freight_over_cap" | "price_deviation" | "other")
          escalation_reason (str)
        """
        inv_total    = context.get("invoice_total", 0.0)
        freight      = context.get("freight_charge", 0.0)
        freight_cap  = context.get("freight_cap", 50.0)
        vtype        = context.get("violation_type", "other")
        reason       = context.get("escalation_reason", "policy review requested")

        if vtype == "freight_over_cap":
            return self._handle_freight_escalation(freight, freight_cap, inv_total)
        elif vtype == "price_deviation":
            return self._handle_price_escalation(inv_total, reason)
        else:
            return self._handle_generic_escalation(inv_total, reason)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_freight_escalation(
        self, freight: float, cap: float, total: float
    ) -> ActorResponse:
        overage = freight - cap

        if self.pre_approved:
            msg = (
                f"[MANAGER] I can confirm this freight charge has my pre-approval. "
                f"The vendor required expedited shipping due to supply chain constraints. "
                f"Freight of ${freight:.2f} (over cap ${cap:.2f} by ${overage:.2f}) "
                f"is approved as a one-time exception. Please proceed with APPROVE_FULL "
                f"for the invoice total of ${total:.2f}."
            )
            verdict = "approved"
        elif self.risk_appetite == "conservative" or total > self.budget_authority:
            msg = (
                f"[MANAGER] I have not pre-approved this freight charge. "
                f"Freight of ${freight:.2f} exceeds our policy cap of ${cap:.2f} by ${overage:.2f}. "
                f"Per policy Section 2, this must be REJECTED. "
                f"Request the vendor to resubmit with freight within the approved limit."
            )
            verdict = "denied"
        elif self.risk_appetite == "aggressive":
            msg = (
                f"[MANAGER] The freight overage of ${overage:.2f} is within acceptable business "
                f"tolerance given our relationship with this vendor. "
                f"I am approving the exception. Proceed with APPROVE_FULL for ${total:.2f}."
            )
            verdict = "approved"
        else:  # moderate — depends on overage magnitude
            if overage / cap < 0.20:
                msg = (
                    f"[MANAGER] The freight overage (${overage:.2f}) is minor — under 20% of cap. "
                    f"I'll approve this as a one-time exception. Proceed with APPROVE_FULL."
                )
                verdict = "approved"
            else:
                msg = (
                    f"[MANAGER] The freight overage of ${overage:.2f} is significant "
                    f"({(overage/cap*100):.0f}% over cap). Without prior written approval, "
                    f"I cannot authorise this. Please REJECT and request a corrected invoice."
                )
                verdict = "denied"

        return ActorResponse(
            actor_type=ActorType.manager,
            persona=ActorPersona.honest,
            message=msg,
            verdict=verdict,
            context_tag=msg,
        )

    def _handle_price_escalation(self, total: float, reason: str) -> ActorResponse:
        if total <= self.budget_authority:
            msg = (
                f"[MANAGER] I've reviewed the escalation: {reason}. "
                f"The total of ${total:.2f} is within my approval authority. "
                f"However, unit price deviations must be corrected before payment. "
                f"Please REJECT and request a corrected invoice at the agreed PO rate."
            )
            verdict = "denied"
        else:
            msg = (
                f"[MANAGER] This escalation (total ${total:.2f}) exceeds my authority. "
                f"Escalating to VP Finance. In the meantime, please HOLD the invoice "
                f"pending senior approval."
            )
            verdict = "escalated_to_vp"

        return ActorResponse(
            actor_type=ActorType.manager,
            persona=ActorPersona.honest,
            message=msg,
            verdict=verdict,
            context_tag=msg,
        )

    def _handle_generic_escalation(self, total: float, reason: str) -> ActorResponse:
        msg = (
            f"[MANAGER] Escalation received: {reason}. "
            f"Invoice total is ${total:.2f}. "
            f"Please follow standard policy — without specific documented exceptions, "
            f"all policy violations must be REJECTED and resubmitted."
        )
        return ActorResponse(
            actor_type=ActorType.manager,
            persona=ActorPersona.honest,
            message=msg,
            verdict="follow_policy",
            context_tag=msg,
        )
