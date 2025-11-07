import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from src.config import db_config
from src.models import Base

def setup_test_data():
    """Setup test data for backend development menggunakan SQLAlchemy"""
    
    # Create engine and connection
    engine = create_engine(db_config.sync_connection_string)
    
    print("🚀 Setting up test data for Sentil Backend (SQLAlchemy)...")
    
    try:
        with engine.begin() as conn:
            # Clear existing test data (dengan urutan yang benar untuk foreign keys)
            conn.execute(text("DELETE FROM output_results"))
            conn.execute(text("DELETE FROM input_queue"))
            conn.execute(text("DELETE FROM training_datasets"))
            conn.execute(text("DELETE FROM system_log"))
            conn.execute(text("DELETE FROM session_slots"))
            conn.execute(text("DELETE FROM users"))
            
            print("✅ Cleared existing test data")
            
            # Create test users
            users_data = [
                {"username": "guest_user", "email": "guest@test.com", "tier": 1},
                {"username": "registered_user", "email": "registered@test.com", "tier": 2},
                {"username": "premium_user", "email": "premium@test.com", "tier": 3}
            ]
            
            user_ids = []
            for user in users_data:
                result = conn.execute(
                    text("""
                    INSERT INTO users (username, email, tier) 
                    VALUES (:username, :email, :tier)
                    RETURNING user_id
                    """),
                    user
                )
                user_id = result.scalar()
                user_ids.append(user_id)
                print(f"✅ Created user: {user['username']} (Tier {user['tier']}) - ID: {user_id}")
            
            # Create session slots (matching your blueprint composition)
            slots_data = []
            # Tier 1: 10 slots (10%)
            slots_data.extend([(1,)] * 10)
            # Tier 2: 30 slots (30%)
            slots_data.extend([(2,)] * 30)
            # Tier 3: 60 slots (60%)
            slots_data.extend([(3,)] * 60)
            
            for tier in slots_data:
                conn.execute(
                    text("INSERT INTO session_slots (tier, is_active) VALUES (:tier, FALSE)"),
                    {"tier": tier[0]}
                )
            
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
                
                conn.execute(
                    text("""
                    INSERT INTO input_queue (user_id, input_text, method, tier, status)
                    VALUES (:user_id, :text, :method, :tier, 'queued')
                    """),
                    {
                        "user_id": user_id,
                        "text": text,
                        "method": method,
                        "tier": user_tier
                    }
                )
                print(f"✅ Created queue item: {text[:30]}... (Tier {user_tier}, {method})")
            
            print("🎉 Test data setup completed!")
            print(f"📊 Created:")
            print(f"   - {len(user_ids)} test users")
            print(f"   - {len(slots_data)} session slots")
            print(f"   - {len(sample_texts)} queue items")
            
    except Exception as e:
        print(f"❌ Error setting up test data: {e}")
        raise

if __name__ == "__main__":
    setup_test_data()
