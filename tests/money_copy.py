"""What counts as a price in a file Blueprint gives a deployment company (owner, 2026-11-12: the site pays nothing;
deployment companies will pay, how is not decided, and no price is stated anywhere).

The same reading as the WebApp's ``scripts/qa/price-statements.ts``, sentence by sentence:

- an amount of money is always a finding ("$0", "USD 30", "30 dollars"); currency codes are matched in capitals only,
  so OpenUSD (``.usd``, ``#usda 1.0``) is never read as money;
- a rate ("per month", "/mo", "per introduction") is always a finding;
- a payment element (checkout, card details, billing, invoice, subscription, Stripe, pricing) and the words fee and
  price are findings unless negated within a few words before;
- one of Blueprint's own spend fields by name (``reserved_cost_usd``, ``max_spend_usd``, ``budget_usd``): internal
  spend tracking is Blueprint's control, never a figure for a company.

Used by ``test_website_packet_facts.py`` and ``test_website_company_simulation.py``.
"""
from __future__ import annotations

import re

_NEGATED_BEFORE = re.compile(r"\b(?:no|not|never|nothing|without|nor|none)\b|n['’]t\b", re.I)
#: Case-sensitive on purpose: ``USD`` in capitals is money, ``usd``/``OpenUSD`` is a file format.
_AMOUNT = re.compile(r"[$€£¥]\s?\d|\b(?:USD|EUR|GBP|CAD|AUD)\s?\$?\d|\d[\d,.]*\s?(?:k|m)?\s?(?:USD|EUR|GBP)\b")
_DOLLAR_WORDS = re.compile(r"\d[\d,.]*\s?(?:dollars?|euros?|cents?)\b", re.I)
_RATE = re.compile(r"\bper\s+(?:month|year|annum|seat|user|introduction|intro|lead|match|referral)\b"
                   r"|/\s?(?:mo|month|yr|year)\b|\b(?:monthly|yearly|annual)\s+(?:fee|plan|price|subscription|charge|billing)\b",
                   re.I)
_PAYMENT = re.compile(r"\b(?:checkout|pay now|buy now|add to cart|credit card|debit card|card number|card details|billing"
                      r"|invoices?|invoiced|subscriptions?|subscribe to (?:a |the |our )?plan|paid plans?|pricing|price list"
                      r"|free trial|stripe|paypal)\b", re.I)
_FEE_OR_PRICE = re.compile(r"\b(?:fees?|prices?|priced)\b", re.I)
_SPEND_FIELD = re.compile(r"\b[a-z_]*(?:cost|spend|spent|price|budget|charge|fee)[a-z_]*_usd\b|\busd_(?:cost|spend|price)\b", re.I)


def _unnegated(sentence: str, pattern: re.Pattern[str]) -> bool:
    return any(not _NEGATED_BEFORE.search(sentence[max(0, match.start() - 60):match.start()])
               for match in pattern.finditer(sentence))


RULES = (
    ("an amount of money", lambda sentence: bool(_AMOUNT.search(sentence) or _DOLLAR_WORDS.search(sentence))),
    ("a rate", lambda sentence: bool(_RATE.search(sentence))),
    ("a payment element", lambda sentence: _unnegated(sentence, _PAYMENT)),
    ("a fee or a price", lambda sentence: _unnegated(sentence, _FEE_OR_PRICE)),
    ("a spend field", lambda sentence: bool(_SPEND_FIELD.search(sentence))),
)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def price_findings(text: str) -> list[str]:
    """Each rule each sentence of ``text`` breaks, as ``rule: sentence``."""
    return [f"{name}: {sentence}" for sentence in sentences(text) for name, test in RULES if test(sentence)]
