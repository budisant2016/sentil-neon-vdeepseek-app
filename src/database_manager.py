from sqlalchemy import create_engine, text, and_, or_, update, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from src.config import db_config
from src.models import User, SessionSlot, InputQueue, OutputResult, SystemLog, TrainingDataset
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # Gunakan connection string dari config
        connection_string = db_config.sqlalchemy_connection_string
        
        if not connection_string:
            raise ValueError("❌ Database connection string tidak ditemukan. Pastikan NEON_DATABASE_URL sudah di-set.")
        
        logger.info(f"🔗 Initializing database connection...")
        
        # Debug info (jangan log password)
        conn_info = db_config.parse_connection_string()
        if conn_info:
            logger.info(f"📡 Connecting to: {conn_info.get('host')}, DB: {conn_info.get('database')}")
        
        try:
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Auto-reconnect
                echo=False  # Set True untuk debug SQL
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("✅ SQLAlchemy database engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database engine: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT version(), current_database(), current_user"))
                row = result.fetchone()
                logger.info(f"✅ Database connected: {row[1]} as {row[2]}")
                return True
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_queued_items(self, limit: int = 10) -> List[Dict]:
        """Get queued items for processing menggunakan SQLAlchemy ORM"""
        with self.get_session() as session:
            try:
                query = (
                    select(
                        InputQueue.queue_id,
                        InputQueue.user_id,
                        InputQueue.input_text,
                        InputQueue.method,
                        InputQueue.tier,
                        SessionSlot.slot_id
                    )
                    .outerjoin(SessionSlot, InputQueue.slot_id == SessionSlot.slot_id)
                    .where(
                        InputQueue.status == 'queued',
                        or_(
                            SessionSlot.is_active.is_(None),
                            SessionSlot.is_active == True
                        )
                    )
                    .order_by(
                        InputQueue.tier,  # Tier 3 first (1,2,3 order)
                        InputQueue.timestamp_in
                    )
                    .limit(limit)
                )
                
                result = session.execute(query)
                items = result.mappings().all()
                logger.info(f"📥 Retrieved {len(items)} queued items")
                return [dict(item) for item in items]
                
            except SQLAlchemyError as e:
                logger.error(f"❌ Failed to get queued items: {e}")
                return []
    
    def update_queue_status(self, queue_id: str, status: str, slot_id: int = None):
        """Update queue item status"""
        with self.get_session() as session:
            try:
                # Get the queue item
                queue_item = session.get(InputQueue, queue_id)
                if queue_item:
                    queue_item.status = status
                    if slot_id is not None:
                        queue_item.slot_id = slot_id
                    session.commit()
                    logger.info(f"✅ Updated queue {queue_id} to status: {status}")
                    return True
                logger.warning(f"⚠️ Queue item {queue_id} not found")
                return False
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"❌ Failed to update queue status: {e}")
                return False
    
    def insert_result(self, queue_id: str, sentiment_label: str, 
                     confidence_score: float, json_result: dict, processed_by: str):
        """Insert analysis result"""
        with self.get_session() as session:
            try:
                result = OutputResult(
                    queue_id=queue_id,
                    sentiment_label=sentiment_label,
                    confidence_score=confidence_score,
                    json_result=json_result,
                    processed_by=processed_by
                )
                session.add(result)
                session.commit()
                logger.info(f"✅ Inserted result for queue {queue_id}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"❌ Failed to insert result: {e}")
                return False
    
    def log_system_activity(self, source: str, message: str, 
                           level: str = 'info', related_id: str = None):
        """Log system activity"""
        with self.get_session() as session:
            try:
                log = SystemLog(
                    source=source,
                    level=level,
                    message=message,
                    related_id=related_id
                )
                session.add(log)
                session.commit()
                logger.debug(f"📝 Logged: {message}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"❌ Failed to log system activity: {e}")
                return False
    
    def acquire_session_slot(self, tier: int, user_id: str) -> Optional[int]:
        """Acquire available session slot for processing"""
        with self.get_session() as session:
            try:
                # Find available slot
                slot_query = (
                    select(SessionSlot.slot_id)
                    .where(
                        SessionSlot.tier == tier,
                        SessionSlot.is_active == False
                    )
                    .limit(1)
                )
                
                slot_result = session.execute(slot_query)
                slot = slot_result.scalar_one_or_none()
                
                if slot:
                    # Update slot to active
                    update_stmt = (
                        update(SessionSlot)
                        .where(SessionSlot.slot_id == slot)
                        .values(
                            is_active=True,
                            current_user_id=user_id,  # ⬅️ FIX: ganti nama
                            started_at=func.now(),
                            expires_at=func.now() + text("INTERVAL '30 minutes'")
                        )
                    )
                    session.execute(update_stmt)
                    session.commit()
                    logger.info(f"✅ Acquired slot {slot} for tier {tier}")
                    return slot
                
                logger.warning(f"⚠️ No available slots for tier {tier}")
                return None
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"❌ Failed to acquire session slot: {e}")
                return None
    
    def release_session_slot(self, slot_id: int):
        """Release session slot after processing"""
        with self.get_session() as session:
            try:
                update_stmt = (
                    update(SessionSlot)
                    .where(SessionSlot.slot_id == slot_id)
                    .values(
                        is_active=False,
                        current_user=None,
                        started_at=None,
                        expires_at=None
                    )
                )
                session.execute(update_stmt)
                session.commit()
                logger.info(f"✅ Released slot {slot_id}")
                return True
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"❌ Failed to release session slot: {e}")
                return False
    
    def get_queue_stats(self) -> List[Dict[str, Any]]:
        """Get queue statistics"""
        with self.get_session() as session:
            try:
                query = (
                    select(
                        InputQueue.status,
                        InputQueue.tier,
                        func.count(InputQueue.queue_id).label('count')
                    )
                    .group_by(InputQueue.status, InputQueue.tier)
                    .order_by(InputQueue.tier, InputQueue.status)
                )
                
                result = session.execute(query)
                stats = result.mappings().all()
                return [dict(stat) for stat in stats]
                
            except SQLAlchemyError as e:
                logger.error(f"❌ Failed to get queue stats: {e}")
                return []
    
    def get_database_info(self):
        """Get database information untuk debug"""
        try:
            with self.get_session() as session:
                # Check tables
                result = session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables = [row[0] for row in result.fetchall()]
                
                # Check queue counts
                queue_result = session.execute(text("""
                    SELECT status, COUNT(*) 
                    FROM input_queue 
                    GROUP BY status
                """))
                queue_stats = dict(queue_result.fetchall())
                
                return {
                    'tables': tables,
                    'queue_stats': queue_stats,
                    'connection_info': db_config.parse_connection_string()
                }
        except Exception as e:
            return {'error': str(e)}

def insert_result(self, queue_id: str, sentiment_label: str, 
                 confidence_score: float, json_result: dict, processed_by: str):
    """Insert analysis result"""
    with self.get_session() as session:
        try:
            result = OutputResult(
                queue_id=queue_id,
                sentiment_label=sentiment_label,
                confidence_score=confidence_score,
                json_result=json_result,
                processed_by=processed_by
            )
            session.add(result)
            session.commit()
            logger.info(f"✅ Inserted result for queue {queue_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"❌ Failed to insert result: {e}")
            return False
