import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker with Indian locale for realistic company names
fake = Faker('en_IN')
np.random.seed(42) # For reproducibility during the hackathon

def generate_synthetic_credit_data(num_records=5000):
    print(f"Generating {num_records} records of synthetic corporate data...")
    
    data = []
    for _ in range(num_records):
        # 1. Generate Base Features
        company_name = fake.company()
        company_age = np.random.randint(1, 50) # Age in years
        cibil_score = np.random.randint(300, 900) # Standard CIBIL Commercial range
        
        # Revenue in Crores (INR) - Skewed towards mid-sized corporates
        gstr_revenue = round(np.random.uniform(5.0, 500.0), 2) 
        
        # Simulate Bank Inflow based on GSTR. 

        # Most are close, but some have massive variance (the "Circular Trading" anomaly)

        variance_factor = np.random.choice(
            [np.random.uniform(0.9, 1.1), np.random.uniform(0.4, 0.7), np.random.uniform(1.3, 1.8)], 
            p=[0.8, 0.1, 0.1] # 80% of companies are honest, 20% have sketchy variance
        )
        bank_inflow = round(gstr_revenue * variance_factor, 2)
        
        # Litigation & Sentiment
        # Most companies have 0 litigation, some have a few
        litigation_count = np.random.choice([0, 1, 2, 3, 5, 10], p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
        # Sentiment from -1.0 (terrible news) to 1.0 (great news)
        news_sentiment = round(np.random.uniform(-1.0, 1.0), 2)

        # 2. Apply "Underwriting" Business Logic to determine Targets
        loan_approved = 1
        rejection_reason = ""
        
        # Rule A: CIBIL Score cutoff
        if cibil_score < 600:
            loan_approved = 0
            
        # Rule B: High Litigation or Terrible News
        if litigation_count >= 3 or news_sentiment < -0.5:
            loan_approved = 0
            
        # Rule C: The "Data Paradox" Check (GST vs Bank mismatch > 25%)
        revenue_variance = abs(gstr_revenue - bank_inflow) / gstr_revenue
        if revenue_variance > 0.25:
            loan_approved = 0
            
        # 3. Calculate Limit and Interest Rate
        if loan_approved == 1:
            # Base limit is roughly 15-25% of their verified bank inflow, scaled by CIBIL
            limit_percentage = np.random.uniform(0.15, 0.25) * (cibil_score / 900)
            approved_limit = round(bank_inflow * limit_percentage, 2)
            
            # Base rate of 8.5%, add risk premium for lower CIBIL or younger companies
            risk_premium = ((900 - cibil_score) / 100) * 0.5
            age_premium = 0 if company_age > 5 else 1.5
            interest_rate = round(8.5 + risk_premium + age_premium, 2)
        else:
            approved_limit = 0.0
            interest_rate = 0.0 # Or use np.nan if you prefer empty values for rejections
            
        # Append row
        data.append({
            "Company_Name": company_name,
            "Company_Age": company_age,
            "CIBIL_Commercial_Score": cibil_score,
            "GSTR_Declared_Revenue_Cr": gstr_revenue,
            "Bank_Statement_Inflow_Cr": bank_inflow,
            "Litigation_Count": litigation_count,
            "News_Sentiment_Score": news_sentiment,
            "Loan_Approved": loan_approved,
            "Approved_Limit_Cr": approved_limit,
            "Interest_Rate_Pct": interest_rate
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Run the function and save to CSV
if __name__ == "__main__":
    synthetic_df = generate_synthetic_credit_data(5000)
    
    # Save to your local directory
    csv_filename = "data.csv"
    synthetic_df.to_csv(csv_filename, index=False)
    
    print(f"Success! {csv_filename} created.")
    print("\nData Snapshot:")
    print(synthetic_df.head())
    
    print("\nApproval Rate Distribution:")
    print(synthetic_df['Loan_Approved'].value_counts(normalize=True) * 100)
