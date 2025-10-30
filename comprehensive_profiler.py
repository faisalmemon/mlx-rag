#!/usr/bin/env python3
"""
Comprehensive profiling script for create_vdb.py
Combines multiple profiling approaches for thorough performance analysis
"""
import cProfile
import pstats
import tracemalloc
import time
import psutil
import os
import sys
from memory_profiler import profile
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vdb import vdb_from_pdf

class PerformanceProfiler:
    def __init__(self, pdf_file="flash_attention.pdf", vdb_file="vdb.npz"):
        self.pdf_file = pdf_file
        self.vdb_file = vdb_file
        self.results = {}
    
    def profile_cprofile(self):
        """Profile using cProfile for function-level timing"""
        print("🔍 Running cProfile analysis...")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run the target function
        start_time = time.time()
        m = vdb_from_pdf(self.pdf_file)
        m.savez(self.vdb_file)
        end_time = time.time()
        
        profiler.disable()
        
        # Save detailed profile
        profiler.dump_stats('profile_cprofile.prof')
        
        # Generate text report
        with open('profile_cprofile.txt', 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats(50)  # Top 50 functions
        
        # Generate focused reports
        with open('profile_mlx_only.txt', 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats('mlx')  # MLX-related functions only
            
        self.results['cprofile'] = {
            'total_time': end_time - start_time,
            'profile_file': 'profile_cprofile.prof'
        }
        
        print(f"✅ cProfile completed in {end_time - start_time:.2f}s")
        return end_time - start_time
    
    def profile_memory(self):
        """Profile memory usage"""
        print("🔍 Running memory analysis...")
        
        # Start memory tracing
        tracemalloc.start()
        process = psutil.Process()
        
        # Memory before
        mem_before = process.memory_info()
        snapshot_before = tracemalloc.take_snapshot()
        
        # Run the target function
        start_time = time.time()
        m = vdb_from_pdf(self.pdf_file)
        m.savez(self.vdb_file)
        end_time = time.time()
        
        # Memory after
        mem_after = process.memory_info()
        snapshot_after = tracemalloc.take_snapshot()
        
        # Calculate differences
        rss_diff = mem_after.rss - mem_before.rss
        vms_diff = mem_after.vms - mem_before.vms
        
        # Top memory allocations
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        
        with open('profile_memory.txt', 'w') as f:
            f.write(f"Memory Usage Analysis\n")
            f.write(f"====================\n\n")
            f.write(f"RSS Memory Change: {rss_diff / 1024 / 1024:.2f} MB\n")
            f.write(f"VMS Memory Change: {vms_diff / 1024 / 1024:.2f} MB\n")
            f.write(f"Peak RSS: {mem_after.rss / 1024 / 1024:.2f} MB\n\n")
            f.write("Top 20 Memory Allocations:\n")
            f.write("-" * 50 + "\n")
            
            for index, stat in enumerate(top_stats[:20], 1):
                f.write(f"{index}. {stat}\n")
        
        tracemalloc.stop()
        
        self.results['memory'] = {
            'rss_change_mb': rss_diff / 1024 / 1024,
            'peak_rss_mb': mem_after.rss / 1024 / 1024,
            'time': end_time - start_time
        }
        
        print(f"✅ Memory analysis completed - Peak: {mem_after.rss / 1024 / 1024:.2f} MB")
        return rss_diff / 1024 / 1024
    
    def profile_step_by_step(self):
        """Profile individual steps in the pipeline"""
        print("🔍 Running step-by-step analysis...")
        
        timings = {}
        
        # Step 1: PDF parsing
        start = time.time()
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(self.pdf_file)
        content = "\n\n".join([e.text for e in elements])
        timings['pdf_parsing'] = time.time() - start
        
        # Step 2: Model initialization
        start = time.time()
        from vdb import VectorDB
        model = VectorDB()
        timings['model_init'] = time.time() - start
        
        # Step 3: Text chunking
        start = time.time()
        from vdb import split_text_into_chunks
        chunks = split_text_into_chunks(text=content, chunk_size=1000, overlap=200)
        timings['text_chunking'] = time.time() - start
        
        # Step 4: Embedding generation
        start = time.time()
        embeddings = model.model.run(chunks)
        timings['embedding_generation'] = time.time() - start
        
        # Step 5: Data preparation and saving
        start = time.time()
        model.embeddings = embeddings
        model.content = chunks
        model.savez(self.vdb_file)
        timings['data_saving'] = time.time() - start
        
        # Save timing report
        with open('profile_steps.txt', 'w') as f:
            f.write("Step-by-Step Performance Analysis\n")
            f.write("=================================\n\n")
            total_time = sum(timings.values())
            f.write(f"Total Time: {total_time:.2f}s\n\n")
            
            for step, duration in timings.items():
                percentage = (duration / total_time) * 100
                f.write(f"{step:20s}: {duration:6.2f}s ({percentage:5.1f}%)\n")
            
            f.write(f"\nAdditional Info:\n")
            f.write(f"PDF Elements: {len(elements)}\n")
            f.write(f"Content Length: {len(content):,} chars\n")
            f.write(f"Number of Chunks: {len(chunks)}\n")
            f.write(f"Embedding Shape: {embeddings.shape}\n")
        
        self.results['step_by_step'] = timings
        print(f"✅ Step-by-step analysis completed - Total: {sum(timings.values()):.2f}s")
        return timings
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        with open('profile_summary.txt', 'w') as f:
            f.write("MLX RAG Performance Profile Summary\n")
            f.write("==================================\n\n")
            f.write(f"Input PDF: {self.pdf_file}\n")
            f.write(f"Output VDB: {self.vdb_file}\n\n")
            
            if 'cprofile' in self.results:
                f.write(f"Total Execution Time: {self.results['cprofile']['total_time']:.2f}s\n")
            
            if 'memory' in self.results:
                f.write(f"Peak Memory Usage: {self.results['memory']['peak_rss_mb']:.2f} MB\n")
                f.write(f"Memory Growth: {self.results['memory']['rss_change_mb']:.2f} MB\n")
            
            if 'step_by_step' in self.results:
                f.write(f"\nPerformance Breakdown:\n")
                timings = self.results['step_by_step']
                total = sum(timings.values())
                for step, duration in timings.items():
                    pct = (duration / total) * 100
                    f.write(f"  {step:20s}: {duration:6.2f}s ({pct:5.1f}%)\n")
            
            f.write(f"\nGenerated Files:\n")
            f.write(f"  - profile_cprofile.prof (detailed function profiling)\n")
            f.write(f"  - profile_cprofile.txt (human-readable cProfile output)\n")
            f.write(f"  - profile_mlx_only.txt (MLX-specific function calls)\n")
            f.write(f"  - profile_memory.txt (memory allocation details)\n")
            f.write(f"  - profile_steps.txt (step-by-step timing breakdown)\n")
            f.write(f"  - profile_summary.txt (this summary)\n")

def main():
    parser = argparse.ArgumentParser(description="Profile create_vdb.py performance")
    parser.add_argument("--pdf", default="flash_attention.pdf", help="PDF file to process")
    parser.add_argument("--vdb", default="vdb.npz", help="Output VDB file")
    parser.add_argument("--skip-cprofile", action="store_true", help="Skip cProfile analysis")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory analysis")
    parser.add_argument("--skip-steps", action="store_true", help="Skip step-by-step analysis")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive performance profiling...")
    print(f"📄 PDF: {args.pdf}")
    print(f"💾 VDB: {args.vdb}")
    print()
    
    profiler = PerformanceProfiler(args.pdf, args.vdb)
    
    try:
        if not args.skip_cprofile:
            profiler.profile_cprofile()
            print()
        
        if not args.skip_memory:
            profiler.profile_memory()
            print()
        
        if not args.skip_steps:
            profiler.profile_step_by_step()
            print()
        
        profiler.generate_summary_report()
        
        print("✅ All profiling completed!")
        print("📊 Check profile_summary.txt for overview")
        print("📁 Detailed reports available in profile_*.txt files")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()