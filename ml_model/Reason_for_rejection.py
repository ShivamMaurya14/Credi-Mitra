"""Credit decision helper for approval, rejection reason, and pricing."""

from typing import Dict, Optional


def evaluate_credit_decision(
    cibil_score: int,
    litigation_count: int,
    news_sentiment: float,
    gstr_revenue: float,
    bank_inflow: float,
    company_age: int,
) -> Dict[str, Optional[float]]:
    """Evaluate core underwriting rules and return decision details."""
    loan_approved = 1
    rejection_reason = ""

    if cibil_score < 600:
        loan_approved = 0
        rejection_reason = "CIBIL Score below cutoff"

    if litigation_count >= 3 or news_sentiment < -0.5:
        loan_approved = 0
        rejection_reason = "High Litigation or Terrible News"

    revenue_variance = abs(gstr_revenue - bank_inflow) / gstr_revenue if gstr_revenue else 1.0
    if revenue_variance > 0.25:
        loan_approved = 0
        rejection_reason = "GST vs Bank Statement Mismatch > 25%"

    if loan_approved == 1:
        limit_base = 25 if cibil_score >= 750 else 15
        limit_percentage = limit_base * (cibil_score / 900)
        approved_limit = round(bank_inflow * limit_percentage, 2)

        base_premium = 8.5
        risk_premium = ((900 - cibil_score) / 100) * 0.5
        age_premium = 0 if company_age > 5 else 1.5
        interest_rate = round(base_premium + risk_premium + age_premium, 2)
    else:
        approved_limit = 0.0
        interest_rate = 0.0

    return {
        "loan_approved": loan_approved,
        "rejection_reason": rejection_reason if not loan_approved else None,
        "approved_limit": approved_limit,
        "interest_rate": interest_rate,
    }
