import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    host: str = os.getenv('NEON_HOST', 'ep-cool-sound-123456.us-east-2.aws.neon.tech')
    database: str = os.getenv('NEON_DB', 'sentil_db')
    user: str = os.getenv('NEON_USER', 'default_user')
    password: str = os.getenv('NEON_PASSWORD', '')
    port: str = os.getenv('NEON_PORT', '5432')
    
    @property
    def connection_string(self):
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_connection_string(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class AppConfig:
    processing_batch_size: int = int(os.getenv('PROCESSING_BATCH_SIZE', 5))
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')

db_config = DatabaseConfig()
app_config = AppConfig()
