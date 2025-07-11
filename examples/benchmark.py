# uhop/examples/benchmark.py
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from uhop import Tensor, Executor
from uhop.core.tuner import KernelTuner

class UHOPBenchmark:
    def __init__(self):
        self.results = {}
        self.monitor = None
        self.tuner = KernelTuner('cuda')
    
    def benchmark_operation(self, operation, sizes, backend, 
                           warmup=3, repeats=5, dtype=np.float32):
        """Benchmark a single operation for given sizes and backend"""
        executor = Executor(backend)
        op_results = {}
        
        for size in sizes:
            # Create input tensors
            if operation == 'matmul':
                a = Tensor(np.random.randn(size, size).astype(dtype))
                b = Tensor(np.random.randn(size, size).astype(dtype))
            else:  # relu or other element-wise ops
                a = Tensor(np.random.randn(size, size).astype(dtype))
                b = None
            
            # Warm-up runs
            for _ in range(warmup):
                if operation == 'matmul':
                    executor.matmul(a, b)
                elif operation == 'relu':
                    executor.relu(a)
            
            # Timed runs
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                
                if operation == 'matmul':
                    result = executor.matmul(a, b)
                elif operation == 'relu':
                    result = executor.relu(a)
                
                # Ensure GPU ops complete
                if backend == 'cuda':
                    import pycuda.driver as cuda
                    cuda.Context.synchronize()
                
                times.append(time.perf_counter() - start)
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_dev = np.std(times)
            
            # Calculate appropriate metrics
            if operation == 'matmul':
                gflops = (2 * size**3) / (avg_time * 1e9)
                metric = gflops
                metric_name = 'gflops'
            else:  # relu
                gbps = (size * size * 4 * 2) / (avg_time * 1e9)  # 2x memory (read+write)
                metric = gbps
                metric_name = 'gbps'
            
            op_results[size] = {
                'avg_time': avg_time,
                'std_dev': std_dev,
                metric_name: metric,
                'times': times
            }
            
            print(f"{backend} {operation} {size}x{size}: "
                  f"{avg_time:.6f}s ± {std_dev:.6f} "
                  f"({metric:.2f} {metric_name.upper()})")
        
        return op_results
    
    def benchmark_tuning(self, sizes, block_sizes=[16, 32, 64, 128]):
        """Benchmark different tuning configurations"""
        tuning_results = {}
        executor = Executor('cuda')
        
        for size in sizes:
            size_results = {}
            a = Tensor(np.random.randn(size, size).astype(np.float32))
            b = Tensor(np.random.randn(size, size).astype(np.float32))
            
            for block_size in block_sizes:
                times = []
                for _ in range(5):  # Multiple runs
                    start = time.perf_counter()
                    
                    # Manually run with specific block size
                    if block_size == 'auto':
                        # Use tuner to find optimal size
                        optimal = self.tuner.tune_matmul(a, b, block_sizes)
                        result = executor.matmul(a, b)
                    else:
                        # Force specific block size
                        from uhop.adapters import cuda_adapter
                        result = cuda_adapter.matmul(a, b, block_size=block_size)
                    
                    import pycuda.driver as cuda
                    cuda.Context.synchronize()
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times)
                gflops = (2 * size**3) / (avg_time * 1e9)
                
                config_name = f"{block_size}x{block_size}" if isinstance(block_size, int) else block_size
                size_results[config_name] = {
                    'avg_time': avg_time,
                    'gflops': gflops
                }
                
                print(f"Tuning {size}x{size} @ {config_name}: "
                      f"{avg_time:.6f}s ({gflops:.2f} GFLOPS)")
            
            # Add auto-tuned result
            self.tuner.tune_matmul(a, b, block_sizes)
            optimal_block = self.tuner.get_optimal_block_size('matmul', a.shape, b.shape)
            size_results['auto-tuned'] = size_results[f"{optimal_block}x{optimal_block}"]
            size_results['auto-tuned']['block_size'] = optimal_block
            
            tuning_results[size] = size_results
        
        return tuning_results
    
    def run_full_benchmark(self, sizes, backends, operations):
        """Run comprehensive benchmark suite"""
        benchmark_start = time.time()
        all_results = {
            'metadata': {
                'system': 'U-HOP v0.2',
                'date': time.ctime(),
                'matrix_sizes': sizes,
                'backends': backends,
                'operations': operations
            },
            'results': {}
        }
        
        # Benchmark operations
        for op in operations:
            all_results['results'][op] = {}
            for backend in backends:
                print(f"\n{'='*50}")
                print(f"=== Benchmarking {op.upper()} on {backend.upper()} ===")
                print(f"{'='*50}")
                all_results['results'][op][backend] = self.benchmark_operation(
                    op, sizes, backend
                )
        
        # Benchmark tuning for matmul on CUDA
        if 'cuda' in backends and 'matmul' in operations:
            print(f"\n{'='*50}")
            print(f"=== Benchmarking MATMUL Tuning on CUDA ===")
            print(f"{'='*50}")
            all_results['results']['matmul_tuning'] = self.benchmark_tuning(sizes)
        
        # Save results
        with open(f'uhop_benchmark_{int(time.time())}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nBenchmark completed in {time.time() - benchmark_start:.2f} seconds")
        return all_results
    
    def plot_results(self, results):
        """Plot all benchmark results with professional formatting"""
        # Create figure grid
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('U-HOP Performance Benchmark Suite', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # Determine subplot layout
        num_ops = len(results['results']) - ('matmul_tuning' in results['results'])
        cols = 2
        rows = (num_ops + 1) // cols + (1 if (num_ops + 1) % cols else 0)
        
        # Plot each operation
        plot_idx = 1
        for op, backend_data in results['results'].items():
            if op == 'matmul_tuning':
                continue
                
            plt.subplot(rows, cols, plot_idx)
            for backend, size_data in backend_data.items():
                sizes = sorted(size_data.keys())
                
                # Extract appropriate metric
                if op == 'matmul':
                    metric = [size_data[size]['gflops'] for size in sizes]
                    ylabel = 'GFLOPS'
                else:  # relu or other
                    metric = [size_data[size]['gbps'] for size in sizes]
                    ylabel = 'GB/s'
                
                plt.plot(sizes, metric, 'o-', linewidth=2.5, 
                         label=f"{backend.upper()}", markersize=8)
            
            plt.title(f'{op.upper()} Performance', fontsize=16)
            plt.xlabel('Matrix Size', fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            plt.xscale('log')
            
            if plot_idx == 1:
                plt.legend(fontsize=12, loc='upper left')
            
            plot_idx += 1
        
        # Plot tuning results
        if 'matmul_tuning' in results['results']:
            plt.subplot(rows, cols, plot_idx)
            tuning_data = results['results']['matmul_tuning']
            
            for size, configs in tuning_data.items():
                config_names = [k for k in configs.keys() if k != 'auto-tuned']
                gflops = [configs[c]['gflops'] for c in config_names]
                
                # Highlight auto-tuned result
                auto_gflops = configs['auto-tuned']['gflops']
                auto_block = configs['auto-tuned']['block_size']
                
                plt.plot(config_names, gflops, 'o-', label=f'{size}x{size}')
                plt.plot('auto-tuned', auto_gflops, 's', markersize=10,
                         label=f'Auto-Tuned ({auto_block}x{auto_block})')
            
            plt.title('MATMUL Kernel Tuning (CUDA)', fontsize=16)
            plt.xlabel('Block Size Configuration', fontsize=12)
            plt.ylabel('GFLOPS', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
        
        # Footer and save
        plt.figtext(0.5, 0.01, 
                    "U-HOP (Universal Hardware Optimization Protocol) | github.com/uhop-project", 
                    ha="center", fontsize=12, fontstyle='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f'uhop_benchmark_{int(time.time())}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run(self):
        """Main benchmark execution"""
        # Benchmark parameters
        sizes = [64, 128, 256, 512, 1024, 2048]
        backends = ['cpu', 'cuda']
        operations = ['matmul', 'relu']
        
        # Run full benchmark suite
        results = self.run_full_benchmark(sizes, backends, operations)
        
        # Plot results
        self.plot_results(results)

if __name__ == "__main__":
    benchmark = UHOPBenchmark()
    benchmark.run()