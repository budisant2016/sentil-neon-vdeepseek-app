import os
from dataclasses import dataclass
from dotenv import load_dotenv
from urllib.parse import urlparse
import streamlit as st

# Try to load from .env file first (for local development)
load_dotenv()

@dataclass
class DatabaseConfig:
    # Priority: Streamlit Secrets > Environment Variable > Default
    def _get_connection_string(self):
        """Get connection string dengan priority order"""
        # 1. Coba dari Streamlit Secrets (Production - Streamlit Cloud)
        try:
            if hasattr(st, 'secrets') and 'connections' in st.secrets:
                neon_secret = st.secrets.connections.get('NEON_DB_URL')
                if neon_secret:
                    return neon_secret
        except:
            pass
        
        # 2. Coba dari Environment Variable (Development)
        env_connection = os.getenv('NEON_DATABASE_URL')
        if env_connection:
            return env_connection
            
        # 3. Fallback
        return ""
    
    @property
    def connection_string(self):
        return self._get_connection_string()
    
    @property
    def sqlalchemy_connection_string(self):
        """Convert untuk SQLAlchemy"""
        conn_str = self.connection_string
        if conn_str:
            # Jika sudah format postgresql://, gunakan langsung
            if conn_str.startswith('postgresql://'):
                return conn_str.replace('postgresql://', 'postgresql+psycopg2://', 1)
            # Jika format postgres:// (legacy), konversi
            elif conn_str.startswith('postgres://'):
                return conn_str.replace('postgres://', 'postgresql+psycopg2://', 1)
        return conn_str
    
    @property
    def sync_connection_string(self):
        """Convert untuk sync connection (setup data)"""
        conn_str = self.connection_string
        if conn_str:
            return conn_str.replace('postgresql+psycopg2://', 'postgresql://', 1)
        return conn_str
    
    def parse_connection_string(self):
        """Parse connection string untuk debug info (without password)"""
        conn_str = self.connection_string
        if not conn_str:
            return {}
        try:
            parsed = urlparse(conn_str)
            return {
                'host': parsed.hostname,
                'database': parsed.path[1:] if parsed.path else '',  # Remove leading /
                'user': parsed.username,
                'port': parsed.port or 5432,
                'ssl_mode': 'require' if 'sslmode=require' in conn_str else 'disabled'
            }
        except Exception as e:
            return {'error': str(e)}

@dataclass
class AppConfig:
    processing_batch_size: int = int(os.getenv('PROCESSING_BATCH_SIZE', 5))
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

db_config = DatabaseConfig()
app_config = AppConfig()
