from typing import Dict, List, Any, Optional, Union
from unittest import result
from venv import logger
from supabase import create_client, Client

class DatabaseManager:
    def __init__(self, logger, project_url, api_key):
        self.logger = logger
        self.supabase_url = project_url
        self.supabase_key = api_key
        self.client: Optional[Client] = None

        if not self.supabase_url or not self.supabase_key:
            self.logger.error("Missing Supabase credentials")
            raise ValueError("Missing Supabase credentials")

    def connect(self) -> bool:
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            return self.client is not None
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        self.client = None

    def insert(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not self.client:
            self.connect()
            
        try:
            result = self.client.table(table_name).insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Insert error: {e}")
            return []
    
    def select(self, 
               table_name: str, 
               columns: str = "*", 
               filters: Optional[Dict[str, Any]] = None,
               limit: Optional[int] = None,
               order_by: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        
        if not self.client:
            self.connect()
        
        try:
            query = self.client.table(table_name).select(columns)
            
            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)
            
            if order_by:
                for column, direction in order_by.items():
                    if direction.lower() == 'asc':
                        query = query.order(column)
                    else:
                        query = query.order(column, desc=True)
            
            if limit:
                query = query.limit(limit)
                
            result = query.execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Select error: {e}")
            return []
    
    def update(self, 
               table_name: str, 
               data: Dict[str, Any], 
               filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        if not self.client:
            self.connect()
        
        try:
            query = self.client.table(table_name).update(data)
            
            for column, value in filters.items():
                query = query.eq(column, value)
                
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Update error: {e}")
            return []
    
    def delete(self, table_name: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.client:
            self.connect()
        
        try:
            query = self.client.table(table_name).delete()
            
            for column, value in filters.items():
                query = query.eq(column, value)
                
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return []
    
    def execute_raw_query(self, query_string: str, values: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.client:
            self.connect()
        
        try:
            result = self.client.rpc(query_string, values or {}).execute()
            return result.data
        except Exception as e:
            logger.error(f"Raw query error: {e}")
            return []