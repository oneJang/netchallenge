from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ROUND_NUM: int = 3
    SERVER: str = '192.168.22.165'
    PORT: str = '8080'
    TIMEOUT: int = 1800
    AVAILABLE_CLIENTS: int = 5
    FRACTION_FIT: float = 0.2


