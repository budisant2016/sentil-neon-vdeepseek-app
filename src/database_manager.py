import psycopg2
import psycopg2.extras
from src.config import db_config
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(db_config.connection_string)
            logger.info("✅ Connected to Neon PostgreSQL database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute SQL query with error handling"""
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            self.conn.rollback()
            return None
    
    def get_queued_items(self, limit: int = 10) -> List[Dict]:
        """Get queued items for processing"""
        query = """
        SELECT q.queue_id, q.user_id, q.input_text, q.method, q.tier, s.slot_id
        FROM input_queue q
        LEFT JOIN session_slots s ON q.slot_id = s.slot_id
        WHERE q.status = 'queued' 
        AND (s.is_active IS NULL OR s.is_active = TRUE)
        ORDER BY 
            CASE q.tier 
                WHEN 3 THEN 1 
                WHEN 2 THEN 2 
                WHEN 1 THEN 3 
            END,
            q.timestamp_in
        LIMIT %s
        """
        results = self.execute_query(query, (limit,), fetch=True)
        return [dict(row) for row in results] if results else []
    
    def update_queue_status(self, queue_id: str, status: str, slot_id: int = None):
        """Update queue item status"""
        query = """
        UPDATE input_queue 
        SET status = %s, last_update = CURRENT_TIMESTAMP,
            slot_id = COALESCE(%s, slot_id)
        WHERE queue_id = %s
        """
        return self.execute_query(query, (status, slot_id, queue_id))
    
    def insert_result(self, queue_id: str, sentiment_label: str, 
                     confidence_score: float, json_result: dict, processed_by: str):
        """Insert analysis result"""
        query = """
        INSERT INTO output_results 
        (queue_id, sentiment_label, confidence_score, json_result, processed_by)
        VALUES (%s, %s, %s, %s, %s)
        """
        return self.execute_query(query, 
            (queue_id, sentiment_label, confidence_score, json_result, processed_by))
    
    def log_system_activity(self, source: str, message: str, 
                           level: str = 'info', related_id: str = None):
        """Log system activity"""
        query = """
        INSERT INTO system_log (source, level, message, related_id)
        VALUES (%s, %s, %s, %s)
        """
        return self.execute_query(query, (source, level, message, related_id))
    
    def acquire_session_slot(self, tier: int, user_id: str) -> Optional[int]:
        """Acquire available session slot for processing"""
        query = """
        SELECT slot_id FROM session_slots 
        WHERE tier = %s AND is_active = FALSE 
        LIMIT 1
        """
        slot = self.execute_query(query, (tier,), fetch=True)
        
        if slot:
            slot_id = slot[0]['slot_id']
            update_query = """
            UPDATE session_slots 
            SET is_active = TRUE, current_user = %s, 
                started_at = CURRENT_TIMESTAMP,
                expires_at = CURRENT_TIMESTAMP + INTERVAL '30 minutes'
            WHERE slot_id = %s
            """
            if self.execute_query(update_query, (user_id, slot_id)):
                return slot_id
        return None
    
    def release_session_slot(self, slot_id: int):
        """Release session slot after processing"""
        query = """
        UPDATE session_slots 
        SET is_active = FALSE, current_user = NULL,
            started_at = NULL, expires_at = NULL
        WHERE slot_id = %s
        """
        return self.execute_query(query, (slot_id,))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        query = """
        SELECT 
            status,
            COUNT(*) as count,
            tier
        FROM input_queue 
        GROUP BY status, tier
        ORDER BY tier, status
        """
        results = self.execute_query(query, fetch=True)
        return [dict(row) for row in results] if results else []
