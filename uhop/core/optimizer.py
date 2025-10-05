# uhop/core/optimization_engine.py
import numpy as np
from .monitor import monitor
from .kernel_db import KernelDB
from .tuner import KernelTuner
from .ai_kernel_generator import AIKernelGenerator

class OptimizationEngine:
    def __init__(self, api_key=None):
        self.kernel_db = KernelDB()
        self.tuner = KernelTuner()
        self.ai_generator = AIKernelGenerator(api_key) if api_key else None
        self.hardware_profiles = {}
        
    def register_hardware(self, device_type, capabilities):
        """Register hardware capabilities"""
        self.hardware_profiles[device_type] = {
            'compute_units': capabilities.get('compute_units', 1),
            'memory_bandwidth': capabilities.get('memory_bandwidth'),
            'supported_ops': capabilities.get('supported_ops', [])
        }
    
    def select_implementation(self, operation, tensor_shapes):
        """Dynamically select best kernel implementation"""
        # 1. Check kernel database
        if best_kernel := self.kernel_db.get_best_kernel(operation, *tensor_shapes):
            return best_kernel
        
        # 2. Try AI-generated kernel if available
        if self.ai_generator:
            constraints = self._generate_constraints(operation, tensor_shapes)
            kernel_code = self.ai_generator.generate_kernel(operation, constraints)
            return kernel_code
        
        # 3. Fallback to tuner-based optimization
        return self.tuner.tune_operation(operation, tensor_shapes)
    
    def _generate_constraints(self, operation, shapes):
        """Generate hardware-aware constraints for AI kernel generation"""
        current_device = self._detect_current_device()
        capabilities = self.hardware_profiles.get(current_device, {})
        
        return {
            'operation': operation,
            'input_shape': shapes[0],
            'output_shape': shapes[1],
            'max_threads_per_block': capabilities.get('max_threads', 1024),
            'shared_memory_size': capabilities.get('shared_mem', 48)  # KB
        }
    
    def _detect_current_device(self):
        """Auto-detect current hardware (simplified)"""
        try:
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device(0).name().lower()
        except:
            return "cpu"