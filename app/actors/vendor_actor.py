"""
VendorActor — Simulated vendor agent for AP Commander.

Replaces pre-generated context notes with a stateful actor that responds
to QUERY_VENDOR actions based on its persona:
  honest     — accurately confirms or corrects discrepancies
  fraudulent — attempts to justify inflated prices / denied duplicates
  confused   — ambiguous, partially-helpful replies

Theme #1 (Halluminate — Multi-Actor Environments)
"""

from __future__ import annotations
import random as _random
from typing import Optional

from ..models import ActorResponse, ActorPersona, ActorType


class VendorActor:
    """
    Simulated vendor that responds to AP Clerk QUERY_VENDOR actions.

    Usage:
        actor = VendorActor(persona="honest", seed=42)
        response = actor.respond(context)
    """

    def __init__(
        self,
        persona: ActorPersona | str = ActorPersona.honest,
        seed: Optional[int] = None,
    ) -> None:
        self.persona = ActorPersona(persona)
        self._rng = _random.Random(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def respond(self, context: dict) -> ActorResponse:
        """
        Generate a vendor response based on episode context.

        context keys used:
          invoice_total  (float)
          po_total       (float)
          invoice_id     (str)
          vendor_name    (str)
          is_duplicate   (bool)
          price_deviation (float, fraction)
        """
        if self.persona == ActorPersona.honest:
            return self._honest_response(context)
        elif self.persona == ActorPersona.fraudulent:
            return self._fraudulent_response(context)
        else:
            return self._confused_response(context)

    # ─────────────────────────────────────────────────────────────────────────
    # Persona implementations
    # ─────────────────────────────────────────────────────────────────────────

    def _honest_response(self, ctx: dict) -> ActorResponse:
        inv_id   = ctx.get("invoice_id", "INV-UNKNOWN")
        inv_amt  = ctx.get("invoice_total", 0.0)
        po_amt   = ctx.get("po_total", 0.0)
        is_dup   = ctx.get("is_duplicate", False)
        dev      = ctx.get("price_deviation", 0.0)

        if is_dup:
            msg = (
                f"We acknowledge your query regarding {inv_id}. "
                "After reviewing our records, this invoice was already submitted and paid "
                "under the same invoice number. Please disregard this duplicate submission. "
                "We apologise for the error."
            )
            verdict = "duplicate_confirmed"
        elif abs(dev) > 0.01:
            pct = abs(dev) * 100
            msg = (
                f"Thank you for flagging invoice {inv_id}. "
                f"We confirm there was a data-entry error — the invoiced price deviated "
                f"{pct:.1f}% from the agreed PO rate. "
                f"A corrected invoice for ${po_amt:,.2f} will be reissued within 2 business days."
            )
            verdict = "error_acknowledged"
        else:
            msg = (
                f"Invoice {inv_id} for ${inv_amt:,.2f} is accurate and matches our records. "
                "All line items correspond to the agreed PO quantities and unit prices. "
                "We confirm this is a valid invoice and request prompt payment."
            )
            verdict = "invoice_confirmed"

        return ActorResponse(
            actor_type=ActorType.vendor,
            persona=self.persona,
            message=msg,
            verdict=verdict,
            context_tag=f"[VENDOR] {msg}",
        )

    def _fraudulent_response(self, ctx: dict) -> ActorResponse:
        inv_id  = ctx.get("invoice_id", "INV-UNKNOWN")
        inv_amt = ctx.get("invoice_total", 0.0)
        is_dup  = ctx.get("is_duplicate", False)
        dev     = ctx.get("price_deviation", 0.0)

        if is_dup:
            excuses = [
                f"Invoice {inv_id} is NOT a duplicate. Our system reissued it due to a "
                "technical error in the previous submission which was never processed on your end. "
                "Please proceed with payment immediately.",
                f"We have no record of {inv_id} being paid. Your accounts payable ledger "
                "may contain an error. This is a fresh invoice and requires immediate payment.",
            ]
            msg     = self._rng.choice(excuses)
            verdict = "duplicate_denied_fraudulently"
        elif abs(dev) > 0.01:
            msg = (
                f"The price on invoice {inv_id} is correct. "
                f"The ${inv_amt:,.2f} total reflects updated market rates and a rush-delivery "
                "surcharge agreed verbally with your procurement team last quarter. "
                "Please approve payment without delay."
            )
            verdict = "price_justified_fraudulently"
        else:
            msg = (
                f"Invoice {inv_id} is fully compliant. Any discrepancies noted in your "
                "system are the result of rounding differences on your side. "
                f"The amount ${inv_amt:,.2f} is correct. Please process immediately."
            )
            verdict = "fraudulent_confirmation"

        return ActorResponse(
            actor_type=ActorType.vendor,
            persona=self.persona,
            message=msg,
            verdict=verdict,
            context_tag=f"[VENDOR] {msg}",
        )

    def _confused_response(self, ctx: dict) -> ActorResponse:
        inv_id = ctx.get("invoice_id", "INV-UNKNOWN")
        replies = [
            f"Hi, regarding invoice {inv_id} — we're looking into it. "
            "Can you confirm which line item you're querying? "
            "Our accounts team is reviewing and will get back to you.",
            f"Thanks for your message about {inv_id}. "
            "We believe everything is in order but are double-checking with our warehouse. "
            "Please allow 3–5 business days for a formal response.",
            f"Invoice {inv_id}: we received your query. "
            "Some of our records are currently being audited so we can't confirm details right now. "
            "We recommend holding payment pending our clarification.",
        ]
        msg = self._rng.choice(replies)
        return ActorResponse(
            actor_type=ActorType.vendor,
            persona=self.persona,
            message=msg,
            verdict="inconclusive",
            context_tag=f"[VENDOR] {msg}",
        )
