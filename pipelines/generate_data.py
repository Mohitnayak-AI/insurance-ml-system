import pandas as pd
import numpy as np
import random

def generate_insurance_data(num_records=10000):
    np.random.seed(42)
    
    data = {
        'claim_id': [f'C{i:05d}' for i in range(1, num_records + 1)],
        'policy_id': [f'P{1000+i}' for i in range(1, num_records + 1)],
        'customer_age': np.random.randint(18, 75, size=num_records),
        'customer_gender': np.random.choice(['M', 'F'], size=num_records),
        'policy_tenure_days': np.random.randint(1, 8000, size=num_records),
        'claim_amount': np.random.randint(500, 350000, size=num_records),
        'claim_type': np.random.choice(['Collision', 'Theft', 'Fire', 'Other'], size=num_records),
        'incident_severity': np.random.choice(['Minor Damage', 'Major Damage', 'Total Loss'], size=num_records),
        'num_previous_claims': np.random.randint(0, 6, size=num_records),
        'days_since_last_claim': np.random.randint(30, 1000, size=num_records),
        'police_report_available': np.random.choice(['Yes', 'No'], size=num_records),
        'vehicle_damage': np.random.choice(['Yes', 'No'], size=num_records),
        'property_damage': np.random.choice(['Yes', 'No'], size=num_records),
    }

    df = pd.DataFrame(data)

    # Injecting "Fraud" logic so the model has something to learn:
    # 1. New policies (<180 days) with high claims (>100k)
    # 2. No police report + Total Loss
    # 3. Frequent claimants
    
    fraud_prob = np.zeros(num_records)
    
    # Logic 1: High amount, short tenure
    fraud_prob += ((df['policy_tenure_days'] < 200) & (df['claim_amount'] > 150000)).astype(float) * 0.4
    
    # Logic 2: Major damage/Total Loss but no police report
    fraud_prob += ((df['incident_severity'] != 'Minor Damage') & (df['police_report_available'] == 'No')).astype(float) * 0.3
    
    # Logic 3: Many previous claims
    fraud_prob += (df['num_previous_claims'] > 3).astype(float) * 0.2
    
    # Baseline noise
    fraud_prob += np.random.random(num_records) * 0.1
    
    # If prob > 0.5, mark as fraud (approx 10-15% of the data)
    df['is_fraud'] = (fraud_prob > 0.55).astype(int)
    
    # Fix 'days_since_last_claim' for those with 0 previous claims
    df.loc[df['num_previous_claims'] == 0, 'days_since_last_claim'] = 999
    
    return df

# Create the data
df_final = generate_insurance_data(10000)
df_final.to_csv(r'/Users/mohitrajnayak/Data Sci Work/Data Sci REPO/Prodection-ready-project/insurance-ml-system/data/raw/insurance_claims_10k.csv', index=False)
print("completed generating data")
# Showing first 5 for the user
# print(df_final.head().to_markdown())
# Saving (simulated for the script display)
# df_final.to_csv('data/raw/insurance_claims_10k.csv', index=False)