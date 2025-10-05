# uhop/core/kernel_db.py
import os
import json
import hashlib
import time
from pathlib import Path

class KernelDB:
    def __init__(self, db_path="uhop_kernel_db.json"):
        self.db_path = db_path
        self.db = self._load_db()
        
    def _load_db(self):
        if not Path(self.db_path).exists():
            return {}
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
            
    def add_kernel(self, operation, constraints, kernel_code, exec_time, error, input_shape, output_shape):
        """Add kernel to database with performance metrics"""
        kernel_hash = hashlib.md5(kernel_code.encode()).hexdigest()
        key = f"{operation}_{kernel_hash}"
        
        self.db[key] = {
            "operation": operation,
            "constraints": constraints,
            "kernel_code": kernel_code,
            "exec_time": exec_time,
            "validation_error": error,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "last_used": time.time(),
            "usage_count": 0
        }
        self.save_db()
        
    def get_best_kernel(self, operation, input_shape, output_shape):
        """Retrieve best kernel for given operation and problem size"""
        best_time = float('inf')
        best_kernel = None
        
        for key, data in self.db.items():
            if data["operation"] != operation:
                continue
                
            # Check for compatible problem size
            if (data["input_shape"] == list(input_shape) and 
                data["output_shape"] == list(output_shape)):
                if data["exec_time"] < best_time:
                    best_time = data["exec_time"]
                    best_kernel = data
        
        return best_kernel
    
    def increment_usage(self, kernel_hash):
        """Track how often kernels are used"""
        key = f"{self.db[kernel_hash]['operation']}_{kernel_hash}"
        if key in self.db:
            self.db[key]["usage_count"] += 1
            self.db[key]["last_used"] = time.time()
            self.save_db()
    
    def get_top_kernels(self, operation, limit=5):
        """Get best performing kernels for an operation"""
        kernels = [data for key, data in self.db.items() 
                  if data["operation"] == operation]
        return sorted(kernels, key=lambda x: x["exec_time"])[:limit]