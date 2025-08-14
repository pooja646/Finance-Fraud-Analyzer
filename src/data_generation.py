import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import time

# Configuration
TARGET_SIZE_GB = 1
ESTIMATED_BYTES_PER_ROW = 500  # Adjust based on your actual data
NUM_CUSTOMERS = 200000
TRANSACTIONS_PER_CUSTOMER = 50
FRAUD_RATE = 0.01
CHUNK_SIZE = 5000  # Customers to process at once
OUTPUT_CSV = 'fraud_transactions_dataset.csv'

# Generate a chunk of customer profiles
def generate_customer_chunk(start_id, end_id):
    
    fake = Faker()
    np.random.seed(42)
    return [{
        'customer_id': f'CUST_{i:08d}',
        'join_date': fake.date_between(start_date='-5y', end_date='-1y'),
        'home_lat': float(fake.latitude()),
        'home_long': float(fake.longitude()),
        'base_amount': np.random.uniform(10, 500),
        'preferred_categories': random.sample([
            'groceries', 'electronics', 'dining',
            'travel', 'utilities', 'entertainment'
        ], k=2),
        'risk_profile': np.random.beta(2, 5),
        'activity_level': np.random.uniform(0.5, 2.0)
    } for i in range(start_id, end_id)]

# Generate transactions for a customer chunk
def generate_transactions(customers, start_txn_id):
    
    fake = Faker()
    transactions = []
    txn_id = start_txn_id
    
    for customer in customers:
        num_trans = max(1, int(TRANSACTIONS_PER_CUSTOMER * customer['activity_level']))
        dates = sorted([fake.date_between(start_date=customer['join_date'], end_date='today') 
                       for _ in range(num_trans)])
        
        for i in range(num_trans):
            trans_date = dates[i]
            trans_time = fake.time()
            timestamp = datetime.combine(trans_date, datetime.strptime(trans_time, '%H:%M:%S').time())
            
            # Determine if fraudulent
            is_fraud = random.random() < (FRAUD_RATE * (1 + customer['risk_profile'] * 5))
            
            if is_fraud:
                amount = customer['base_amount'] * random.choice([0.1, 5, 10, 20])
                location = (float(fake.latitude()), float(fake.longitude()))
                category = random.choice([c for c in ['groceries', 'electronics', 'dining', 'travel', 'utilities', 'entertainment'] 
                                       if c not in customer['preferred_categories']])
                device = random.choice(['mobile', 'desktop', None])
                trans_type = random.choice(['online', 'card_not_present'])
                foreign = random.random() > 0.7
            else:
                amount = customer['base_amount'] * np.random.lognormal(0, 0.2)
                location = (customer['home_lat'] + np.random.uniform(-0.5, 0.5),
                           customer['home_long'] + np.random.uniform(-0.5, 0.5))
                category = random.choice(customer['preferred_categories'])
                device = random.choice(['mobile', 'desktop'])
                trans_type = random.choice(['online', 'pos', 'atm'])
                foreign = random.random() > 0.95
            
            time_since_last = np.random.exponential(scale=10) if i == 0 else (
                timestamp - datetime.combine(dates[i-1], datetime.strptime(fake.time(), '%H:%M:%S').time())).total_seconds()/3600
            
            transactions.append({
                'transaction_id': f'TXN_{txn_id:012d}',
                'timestamp': timestamp,
                'customer_id': customer['customer_id'],
                'amount': round(amount, 2),
                'currency': 'USD',
                'merchant_category': category,
                'location_lat': location[0],
                'location_long': location[1],
                'device_type': device,
                'transaction_type': trans_type,
                'is_fraud': int(is_fraud),
                'time_since_last': time_since_last,
                'foreign_country': int(foreign),
                'multiple_quick_trans': int(is_fraud and random.random() > 0.5)
            })
            txn_id += 1
    
    return transactions, txn_id

# Generate and save 1GB CSV file incrementally
def generate_csv():
    
    start_time = time.time()
    total_transactions = 0
    txn_id = 0
    
    # Initialize CSV file with header
    pd.DataFrame(columns=[
        'transaction_id', 'timestamp', 'customer_id', 'amount', 'currency',
        'merchant_category', 'location_lat', 'location_long', 'device_type',
        'transaction_type', 'is_fraud', 'time_since_last', 'foreign_country',
        'multiple_quick_trans'
    ]).to_csv(OUTPUT_CSV, index=False)
    
    # Process in chunks to avoid memory issues
    for chunk_start in range(0, NUM_CUSTOMERS, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, NUM_CUSTOMERS)
        
        # Generate customer chunk
        customers = generate_customer_chunk(chunk_start, chunk_end)
        
        # Generate transactions
        transactions, txn_id = generate_transactions(customers, txn_id)
        total_transactions += len(transactions)
        
        # Append to CSV
        pd.DataFrame(transactions).to_csv(
            OUTPUT_CSV,
            mode='a',
            header=False,
            index=False
        )
        
        # Check file size
        file_size = os.path.getsize(OUTPUT_CSV) / (1024 ** 3)
        print(f"Processed customers {chunk_start}-{chunk_end-1} | "
              f"Transactions: {total_transactions:,} | "
              f"File size: {file_size:.2f}GB", end='\r')
        
        # Stop when we reach target size
        if file_size >= TARGET_SIZE_GB:
            break
    
    # Final stats
    file_size = os.path.getsize(OUTPUT_CSV) / (1024 ** 3)
    print(f"\n\nGenerated {total_transactions:,} transactions")
    print(f"Final file size: {file_size:.2f}GB")
    print(f"Time taken: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    generate_csv()