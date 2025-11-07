from sqlalchemy import Column, String, Integer, Boolean, Float, Text, DateTime, ForeignKey, CheckConstraint, SmallInteger
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), nullable=False)
    email = Column(String(150))
    tier = Column(SmallInteger, default=1)
    registered_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint('tier IN (1,2,3)', name='check_user_tier'),
    )

class SessionSlot(Base):
    __tablename__ = 'session_slots'
    
    slot_id = Column(Integer, primary_key=True, autoincrement=True)
    tier = Column(SmallInteger, nullable=False)
    is_active = Column(Boolean, default=True)
    current_user = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    started_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        CheckConstraint('tier IN (1,2,3)', name='check_slot_tier'),
    )

class InputQueue(Base):
    __tablename__ = 'input_queue'
    
    queue_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    input_text = Column(Text, nullable=False)
    method = Column(String(50), default='NaiveBayes')
    tier = Column(SmallInteger, default=1)
    status = Column(String(20), default='queued')
    slot_id = Column(Integer, ForeignKey('session_slots.slot_id'))
    timestamp_in = Column(DateTime(timezone=True), server_default=func.now())
    last_update = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        CheckConstraint('status IN (\'queued\',\'processing\',\'done\',\'error\')', name='check_queue_status'),
        CheckConstraint('tier IN (1,2,3)', name='check_queue_tier'),
    )

class OutputResult(Base):
    __tablename__ = 'output_results'
    
    result_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    queue_id = Column(UUID(as_uuid=True), ForeignKey('input_queue.queue_id', ondelete='CASCADE'))
    sentiment_label = Column(String(20))
    confidence_score = Column(Float)
    json_result = Column(JSONB)
    processed_by = Column(String(50))
    timestamp_out = Column(DateTime(timezone=True), server_default=func.now())

class TrainingDataset(Base):
    __tablename__ = 'training_datasets'
    
    dataset_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    dataset_name = Column(String(100))
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    file_path = Column(Text)
    size_kb = Column(Integer)

class SystemLog(Base):
    __tablename__ = 'system_log'
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    level = Column(String(10), default='info')
    message = Column(Text, nullable=False)
    related_id = Column(UUID(as_uuid=True))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint('level IN (\'info\',\'warning\',\'error\')', name='check_log_level'),
    )
