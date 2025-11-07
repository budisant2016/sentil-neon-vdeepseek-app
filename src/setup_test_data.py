import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_manager import DatabaseManager
import uuid

def setup_test_data():
    """Setup test data for backend development"""
    db = DatabaseManager()
    
    print("🚀 Setting up test data for Sentil Backend...")
    
    # Clear existing test data
    db.execute_query("DELETE FROM output_results")
    db.execute_query("DELETE FROM input_queue")
    db.execute_query("DELETE FROM session_slots")
    db.execute_query("DELETE FROM users")
    
    # Create test users
    test_users = [
        ('guest_user', 'guest@test.com', 1),
        ('registered_user', 'registered@test.com', 2),
        ('premium_user', 'premium@test.com', 3)
    ]
    
    user_ids = []
    
    for username, email, tier in test_users:
        query = """
        INSERT INTO users (username, email, tier) 
        VALUES (%s, %s, %s)
        RETURNING user_id
        """
        result = db.execute_query(query, (username, email, tier), fetch=True)
        if result:
            user_ids.append(result[0]['user_id'])
            print(f"✅ Created user: {username} (Tier {tier})")
    
    # Create session slots (matching your blueprint composition)
    slots_data = []
    # Tier 1: 10 slots (10%)
    slots_data.extend([(1,)] * 10)
    # Tier 2: 30 slots (30%)
    slots_data.extend([(2,)] * 30)
    # Tier 3: 60 slots (60%)
    slots_data.extend([(3,)] * 60)
    
    for tier in slots_data:
        query = "INSERT INTO session_slots (tier, is_active) VALUES (%s, FALSE)"
        db.execute_query(query, tier)
    
    print(f"✅ Created {len(slots_data)} session slots")
    
    # Create sample queue items
    sample_texts = [
        "I absolutely love this service! It's fantastic!",
        "This is the worst experience ever. Terrible!",
        "The product is okay, but could be better.",
        "Outstanding quality and fast delivery! Wonderful!",
        "Very disappointed with customer support. Poor service.",
        "Good value for the price. Satisfied.",
        "Terrible quality, don't buy this. Waste of money.",
        "Excellent product, highly recommended! Amazing!",
        "Not bad, but not great either. Average.",
        "Amazing features and great support team! Perfect!"
    ]
    
    methods = ['NaiveBayes', 'KNN', 'RandomForest', 'SVM']
    
    for i, text in enumerate(sample_texts):
        user_tier = (i % 3) + 1  # Distribute across tiers
        user_id = user_ids[user_tier - 1]
        method = methods[i % len(methods)]
        
        query = """
        INSERT INTO input_queue (user_id, input_text, method, tier, status)
        VALUES (%s, %s, %s, %s, 'queued')
        """
        db.execute_query(query, (user_id, text, method, user_tier))
        print(f"✅ Created queue item: {text[:30]}... (Tier {user_tier}, {method})")
    
    print("🎉 Test data setup completed!")
    print(f"📊 Created:")
    print(f"   - {len(user_ids)} test users")
    print(f"   - {len(slots_data)} session slots")
    print(f"   - {len(sample_texts)} queue items")

if __name__ == "__main__":
    setup_test_data()
