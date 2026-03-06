loan_approved = 1

rejection_reason = ""
        
# Rule A: CIBIL Score cutoff
if cibil_score < 600:
    loan_approved = 0
    rejection_reason = "CIBIL Score below cutoff"


# Rule B: High Litigation or Terrible News
if litigation_count >= 3 or news_sentiment < -0.5:
    loan_approved = 0
    rejection_reason = "High Litigation or Terrible News"


# Rule C: The "Data Paradox" Check (GST vs Bank mismatch > 25%)
revenue_variance = abs(gstr_revenue - bank_inflow) / gstr_revenue
if revenue_variance > 0.25:
    loan_approved = 0
    rejection_reason = "GST vs Bank Statement Mismatch > 25%"


# 3. Calculate Limit and Interest Rate

if loan_approved == 1:

    # Base limit is roughly 15-25% of their verified bank inflow, scaled by CIBIL
    x = 25 if cibil_score >= 750 else 15
    limit_percentage = x * (cibil_score / 900)
    approved_limit = round(bank_inflow * limit_percentage, 2)


    # Base rate of 8.5%, we can have based of our reserch  + add risk premium for lower CIBIL or younger companies
    Base_premium = 8.5
    risk_premium = ((900 - cibil_score) / 100) * 0.5
    age_premium = 0 if company_age > 5 else 1.5
    interest_rate = round(Base_premium + risk_premium + age_premium, 2)

else:
    approved_limit = 0.0
    interest_rate = 0.0 # Or use np.nan if you prefer empty values for rejection

