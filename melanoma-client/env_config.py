from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EDGE_NUM: str = '3'
    SERVER: str = '192.168.22.165' 
    PORT: str = '8080'
