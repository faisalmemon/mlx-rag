"""
Flash Attention RAG Evaluation Dataset

This dataset contains carefully crafted question-answer pairs for evaluating
RAG system improvements on the Flash Attention paper.

Each test case includes:
- question: The query to test
- expected_answer: What a good RAG system should retrieve/generate
- relevant_sections: Which paper sections contain the answer
- difficulty: easy/medium/hard based on specificity and reasoning required
- answer_type: factual/conceptual/comparative/numerical
"""

import json
from typing import List, Dict, Any

# Test dataset for Flash Attention paper evaluation
FLASH_ATTENTION_TEST_CASES = [
    # === EASY FACTUAL QUESTIONS ===
    {
        "id": "fa_001",
        "question": "What is the main memory bottleneck in standard attention computation?",
        "expected_answer": "HBM (High Bandwidth Memory) access is slow compared to on-chip SRAM memory. Standard attention reads and writes attention matrices to HBM, creating a memory bottleneck.",
        "relevant_sections": ["2.1", "3.1"],
        "difficulty": "easy",
        "answer_type": "factual",
        "keywords": ["HBM", "SRAM", "memory bottleneck", "attention matrices"]
    },
    
    {
        "id": "fa_002", 
        "question": "What does Flash Attention stand for?",
        "expected_answer": "Flash Attention is named for its use of tiling and recomputation to make attention computation 'flash' between different levels of memory hierarchy (SRAM vs HBM).",
        "relevant_sections": ["1", "4.1"],
        "difficulty": "easy",
        "answer_type": "factual",
        "keywords": ["tiling", "recomputation", "memory hierarchy"]
    },

    {
        "id": "fa_003",
        "question": "What are the two main techniques used in Flash Attention?",
        "expected_answer": "Tiling and recomputation. Tiling divides the attention computation into blocks that fit in SRAM. Recomputation avoids storing intermediate attention matrices by recomputing them when needed.",
        "relevant_sections": ["4.1", "4.2"],
        "difficulty": "easy", 
        "answer_type": "factual",
        "keywords": ["tiling", "recomputation", "blocks", "SRAM"]
    },

    # === MEDIUM CONCEPTUAL QUESTIONS ===
    {
        "id": "fa_004",
        "question": "How does Flash Attention achieve memory efficiency without changing the attention output?",
        "expected_answer": "Flash Attention uses mathematically equivalent operations but changes the order of computation. It uses tiling to work on smaller blocks and recomputes attention weights instead of storing them, maintaining numerical equivalence to standard attention.",
        "relevant_sections": ["4.1", "4.2", "4.3"],
        "difficulty": "medium",
        "answer_type": "conceptual",
        "keywords": ["mathematically equivalent", "order of computation", "numerical equivalence"]
    },

    {
        "id": "fa_005",
        "question": "Why is recomputation beneficial in Flash Attention despite doing more work?",
        "expected_answer": "Recomputation trades compute for memory bandwidth. While it requires more FLOPs, it avoids slow HBM reads/writes by keeping data in fast SRAM. Modern GPUs are often memory-bound rather than compute-bound, making this trade-off beneficial.",
        "relevant_sections": ["4.2", "5.1"],
        "difficulty": "medium",
        "answer_type": "conceptual", 
        "keywords": ["compute for memory", "FLOPs", "memory-bound", "compute-bound"]
    },

    {
        "id": "fa_006",
        "question": "What is the IO complexity improvement of Flash Attention?",
        "expected_answer": "Flash Attention reduces IO complexity from O(N²) to O(N²d/M) where N is sequence length, d is head dimension, and M is SRAM size. This is a significant improvement for long sequences.",
        "relevant_sections": ["4.3", "theorem 1"],
        "difficulty": "medium",
        "answer_type": "numerical",
        "keywords": ["IO complexity", "O(N²)", "O(N²d/M)", "sequence length"]
    },

    # === HARD ANALYTICAL QUESTIONS ===
    {
        "id": "fa_007",
        "question": "How does Flash Attention handle the softmax operation across tiles?",
        "expected_answer": "Flash Attention uses online softmax with running maximum and sum tracking. It maintains running statistics across tiles and renormalizes when processing each new tile, ensuring mathematically equivalent results to computing softmax over the full attention matrix.",
        "relevant_sections": ["4.1", "algorithm 1"],
        "difficulty": "hard",
        "answer_type": "conceptual",
        "keywords": ["online softmax", "running maximum", "renormalization", "running statistics"]
    },

    {
        "id": "fa_008",
        "question": "What is the relationship between SRAM size and Flash Attention's efficiency gains?",
        "expected_answer": "Larger SRAM allows bigger tiles, reducing the number of tiles needed and thus reducing recomputation overhead. The efficiency gain is roughly proportional to M/d where M is SRAM size and d is head dimension.",
        "relevant_sections": ["4.3", "5.2"],
        "difficulty": "hard",
        "answer_type": "analytical",
        "keywords": ["SRAM size", "tile size", "recomputation overhead", "proportional"]
    },

    {
        "id": "fa_009",
        "question": "How does Flash Attention's performance scale with sequence length compared to standard attention?",
        "expected_answer": "Standard attention has O(N²) memory complexity making it impractical for long sequences. Flash Attention maintains O(N) memory complexity while achieving 2-4x speedup on sequences up to 2048 tokens, with even larger improvements on longer sequences.",
        "relevant_sections": ["5.1", "5.2", "figure 1"],
        "difficulty": "hard",
        "answer_type": "comparative",
        "keywords": ["sequence length scaling", "O(N²)", "O(N)", "2-4x speedup"]
    },

    # === IMPLEMENTATION SPECIFIC ===
    {
        "id": "fa_010",
        "question": "What GPU memory hierarchy does Flash Attention optimize for?",
        "expected_answer": "Flash Attention optimizes for the two-level GPU memory hierarchy: fast on-chip SRAM (shared memory) and slower off-chip HBM (global memory). It keeps frequently accessed data in SRAM and minimizes HBM transfers.",
        "relevant_sections": ["2.1", "3.1"],
        "difficulty": "medium",
        "answer_type": "factual",
        "keywords": ["GPU memory hierarchy", "SRAM", "HBM", "shared memory", "global memory"]
    },

    {
        "id": "fa_011", 
        "question": "What are the backward pass optimizations in Flash Attention?",
        "expected_answer": "The backward pass also uses tiling and recomputation. It recomputes forward pass intermediate values during backpropagation instead of storing them, maintaining the same memory efficiency benefits for gradient computation.",
        "relevant_sections": ["4.4", "algorithm 2"],
        "difficulty": "hard",
        "answer_type": "conceptual",
        "keywords": ["backward pass", "gradient computation", "recomputes forward pass"]
    },

    # === COMPARATIVE QUESTIONS ===
    {
        "id": "fa_012",
        "question": "How does Flash Attention compare to other memory-efficient attention methods?",
        "expected_answer": "Unlike approximation methods (sparse attention, low-rank), Flash Attention produces exact attention outputs. Unlike gradient checkpointing which trades memory for compute in backprop, Flash Attention optimizes both forward and backward passes through IO-aware algorithms.",
        "relevant_sections": ["6", "related work"],
        "difficulty": "hard", 
        "answer_type": "comparative",
        "keywords": ["exact attention", "sparse attention", "low-rank", "gradient checkpointing"]
    },

    # === PRACTICAL APPLICATION ===
    {
        "id": "fa_013",
        "question": "What types of models benefit most from Flash Attention?",
        "expected_answer": "Models with long sequences benefit most: language models processing long documents, vision transformers with high-resolution images, and any transformer architecture where attention computation becomes a memory bottleneck.",
        "relevant_sections": ["5.3", "applications"],
        "difficulty": "medium",
        "answer_type": "practical",
        "keywords": ["long sequences", "language models", "vision transformers", "memory bottleneck"]
    },

    # === NUMERICAL/QUANTITATIVE ===
    {
        "id": "fa_014",
        "question": "What speedup does Flash Attention achieve on GPUs?",
        "expected_answer": "Flash Attention achieves 2-4x speedup on A100 GPUs for sequence lengths up to 2K tokens, with larger improvements for longer sequences. Wall-clock time improvements are most significant for memory-bound workloads.",
        "relevant_sections": ["5.1", "5.2", "experiments"],
        "difficulty": "medium",
        "answer_type": "numerical", 
        "keywords": ["2-4x speedup", "A100", "2K tokens", "wall-clock time"]
    },

    # === THEORETICAL FOUNDATION ===
    {
        "id": "fa_015",
        "question": "What is the theoretical foundation that ensures Flash Attention produces identical outputs?",
        "expected_answer": "Flash Attention relies on the associativity of matrix multiplication and properties of softmax. The key insight is that attention can be computed in any order as long as the softmax normalization is handled correctly across blocks.",
        "relevant_sections": ["4.1", "theorem 1", "mathematical foundation"],
        "difficulty": "hard",
        "answer_type": "theoretical",
        "keywords": ["associativity", "matrix multiplication", "softmax normalization", "mathematical foundation"]
    }
]


class RAGEvaluator:
    """Evaluation framework for RAG systems using the Flash Attention test dataset."""
    
    def __init__(self, test_cases: List[Dict[str, Any]] = None):
        self.test_cases = test_cases or FLASH_ATTENTION_TEST_CASES
        
    def get_test_cases_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """Filter test cases by difficulty level."""
        return [case for case in self.test_cases if case["difficulty"] == difficulty]
    
    def get_test_cases_by_type(self, answer_type: str) -> List[Dict[str, Any]]:
        """Filter test cases by answer type."""
        return [case for case in self.test_cases if case["answer_type"] == answer_type]
    
    def evaluate_retrieval_hit_rate(self, vdb, k: int = 5) -> Dict[str, float]:
        """
        Evaluate retrieval quality by checking if relevant sections appear in top-k results.
        This is a simple proxy - in practice you'd check actual chunk content.
        """
        results = {
            "overall_hit_rate": 0.0,
            "easy_hit_rate": 0.0, 
            "medium_hit_rate": 0.0,
            "hard_hit_rate": 0.0
        }
        
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
        difficulty_hits = {"easy": 0, "medium": 0, "hard": 0}
        
        total_hits = 0
        
        for case in self.test_cases:
            difficulty_counts[case["difficulty"]] += 1
            
            # In a real implementation, you'd:
            # 1. Query the VDB with case["question"] 
            # 2. Get top-k chunks
            # 3. Check if any chunk contains keywords from case["relevant_sections"]
            # 4. For now, we'll simulate this
            
            # Placeholder for actual retrieval logic
            # hit = self._check_retrieval_hit(vdb, case, k)
            # if hit:
            #     total_hits += 1
            #     difficulty_hits[case["difficulty"]] += 1
        
        # Calculate hit rates (placeholder calculation)
        results["overall_hit_rate"] = total_hits / len(self.test_cases)
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty_counts[difficulty] > 0:
                results[f"{difficulty}_hit_rate"] = difficulty_hits[difficulty] / difficulty_counts[difficulty]
        
        return results
    
    def export_to_json(self, filename: str = "flash_attention_test_dataset.json"):
        """Export test dataset to JSON for external tools."""
        with open(filename, 'w') as f:
            json.dump(self.test_cases, f, indent=2)
        print(f"Test dataset exported to {filename}")
    
    def print_dataset_summary(self):
        """Print summary statistics of the test dataset."""
        total = len(self.test_cases)
        by_difficulty = {}
        by_type = {}
        
        for case in self.test_cases:
            difficulty = case["difficulty"]
            answer_type = case["answer_type"]
            
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
            by_type[answer_type] = by_type.get(answer_type, 0) + 1
        
        print(f"Flash Attention RAG Test Dataset Summary")
        print(f"Total test cases: {total}")
        print(f"\nBy difficulty:")
        for difficulty, count in by_difficulty.items():
            print(f"  {difficulty}: {count} ({count/total*100:.1f}%)")
        print(f"\nBy answer type:")
        for answer_type, count in by_type.items():
            print(f"  {answer_type}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    evaluator.print_dataset_summary()
    evaluator.export_to_json()
    
    # Show a few example test cases
    print(f"\nExample test cases:")
    for i, case in enumerate(FLASH_ATTENTION_TEST_CASES[:3]):
        print(f"\n{i+1}. {case['question']}")
        print(f"   Difficulty: {case['difficulty']}, Type: {case['answer_type']}")
        print(f"   Expected: {case['expected_answer'][:100]}...")