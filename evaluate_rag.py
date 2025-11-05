"""
RAG Evaluation Script for Flash Attention Paper

This script evaluates RAG system performance using the test dataset.
It provides concrete metrics to measure improvements from your research ideas.
"""

import mlx.core as mx
import json
import time
from typing import Dict, List, Tuple, Any

# Import only what we need from vdb to avoid unstructured dependency issues
import sys
sys.path.append('.')

# Simple VectorDB class for evaluation without PDF processing
class SimpleVectorDB:
    def __init__(self, vdb_file: str):
        from model import Model
        self.model = Model()
        self.embeddings = None
        self.content = None
        
        try:
            vdb = mx.load(vdb_file)
            self.embeddings = vdb["embeddings"]
            chunk_data = vdb["chunk_data"]
            chunk_lengths = vdb["chunk_lengths"]
            self.content = self._mx_array_to_chunks(chunk_data, chunk_lengths)
        except Exception as e:
            raise Exception(f"Failed to load VDB: {e}")
    
    def _mx_array_to_chunks(self, data: mx.array, lengths: mx.array) -> List[str]:
        i = 0
        output = []
        for l in lengths:
            j = l.item() + i
            x = [chr(d.item()) for d in data[i:j]]
            output.append("".join(x))
            i = l.item()
        return output
    
    def query(self, text: str) -> str:
        query_emb = self.model.run([text])  # Model expects list of strings
        scores = mx.matmul(query_emb, self.embeddings.T) * 100
        response = self.content[mx.argmax(scores).item()]
        return response

from test_dataset import RAGEvaluator, FLASH_ATTENTION_TEST_CASES


class FlashAttentionRAGEvaluator:
    """Comprehensive evaluation of RAG improvements for Flash Attention paper."""
    
    def __init__(self, vdb_file: str = "vdb.npz"):
        self.vdb_file = vdb_file
        self.test_cases = FLASH_ATTENTION_TEST_CASES
        self.evaluator = RAGEvaluator()
    
    def evaluate_baseline(self) -> Dict[str, Any]:
        """Evaluate current RAG system performance."""
        print("Evaluating baseline RAG system...")
        
        # Load existing vector database
        vdb = SimpleVectorDB(self.vdb_file)
        
        results = {
            "similarity_stats": self._compute_similarity_stats(vdb),
            "retrieval_quality": self._evaluate_retrieval_quality(vdb),
            "response_quality": self._evaluate_response_quality(vdb),
            "performance_metrics": self._measure_performance(vdb)
        }
        
        return results
    
    def _compute_similarity_stats(self, vdb: SimpleVectorDB) -> Dict[str, Any]:
        """Compute the similarity statistics you discovered."""
        if vdb.embeddings is None:
            return {"error": "No embeddings found"}
        
        # Compute pairwise similarities
        similarities = mx.matmul(vdb.embeddings, vdb.embeddings.T)
        
        # Remove diagonal (self-similarities)
        n = similarities.shape[0]
        mask = mx.eye(n, dtype=mx.bool_)
        off_diagonal = mx.where(mask, mx.array(0.0), similarities)
        
        # Compute statistics
        sim_mean = mx.mean(off_diagonal)
        sim_var = mx.var(off_diagonal)
        sim_std = mx.sqrt(sim_var)
        cv = sim_std / sim_mean
        
        return {
            "mean_similarity": float(sim_mean.item()),
            "std_similarity": float(sim_std.item()),
            "coefficient_of_variation": float(cv.item()),
            "min_similarity": float(mx.min(off_diagonal).item()),
            "max_similarity": float(mx.max(off_diagonal).item())
        }
    
    def _evaluate_retrieval_quality(self, vdb: SimpleVectorDB) -> Dict[str, Any]:
        """Evaluate how well the system retrieves relevant chunks."""
        results = {
            "hit_rate_top1": 0.0,
            "hit_rate_top3": 0.0,
            "hit_rate_top5": 0.0,
            "by_difficulty": {},
            "by_type": {},
            "failed_cases": []
        }
        
        total_cases = len(self.test_cases)
        hits_top1 = 0
        hits_top3 = 0
        hits_top5 = 0
        
        difficulty_stats = {"easy": {"total": 0, "hits": 0}, 
                           "medium": {"total": 0, "hits": 0},
                           "hard": {"total": 0, "hits": 0}}
        
        for case in self.test_cases:
            # Get the retrieved response
            try:
                response = vdb.query(case["question"])
                
                # Check if response contains relevant keywords
                # This is a simple heuristic - in practice you'd use more sophisticated matching
                relevant_keywords = case.get("keywords", [])
                response_lower = response.lower()
                
                keyword_matches = sum(1 for keyword in relevant_keywords 
                                    if keyword.lower() in response_lower)
                
                # Consider it a hit if at least 30% of keywords are found
                hit_threshold = max(1, len(relevant_keywords) * 0.3)
                is_hit = keyword_matches >= hit_threshold
                
                if is_hit:
                    hits_top1 += 1
                    hits_top3 += 1  # Top-1 hit means top-3 and top-5 hits too
                    hits_top5 += 1
                    difficulty_stats[case["difficulty"]]["hits"] += 1
                else:
                    results["failed_cases"].append({
                        "id": case["id"],
                        "question": case["question"],
                        "expected_keywords": relevant_keywords,
                        "response_snippet": response[:200] + "..."
                    })
                
                difficulty_stats[case["difficulty"]]["total"] += 1
                
            except Exception as e:
                print(f"Error evaluating case {case['id']}: {e}")
                results["failed_cases"].append({
                    "id": case["id"], 
                    "question": case["question"],
                    "error": str(e)
                })
        
        # Calculate hit rates
        results["hit_rate_top1"] = hits_top1 / total_cases
        results["hit_rate_top3"] = hits_top3 / total_cases  # Would be different with top-k retrieval
        results["hit_rate_top5"] = hits_top5 / total_cases
        
        # Calculate by difficulty
        for difficulty, stats in difficulty_stats.items():
            if stats["total"] > 0:
                results["by_difficulty"][difficulty] = stats["hits"] / stats["total"]
            else:
                results["by_difficulty"][difficulty] = 0.0
        
        return results
    
    def _evaluate_response_quality(self, vdb: SimpleVectorDB) -> Dict[str, Any]:
        """Evaluate the quality of generated responses."""
        results = {
            "avg_response_length": 0.0,
            "responses_with_keywords": 0.0,
            "sample_responses": []
        }
        
        total_length = 0
        keyword_matches = 0
        
        # Sample a few cases for manual inspection
        sample_cases = self.test_cases[:5]
        
        for case in sample_cases:
            try:
                response = vdb.query(case["question"])
                total_length += len(response)
                
                # Check keyword presence
                relevant_keywords = case.get("keywords", [])
                response_lower = response.lower()
                found_keywords = [kw for kw in relevant_keywords if kw.lower() in response_lower]
                
                if found_keywords:
                    keyword_matches += 1
                
                results["sample_responses"].append({
                    "question": case["question"],
                    "response": response,
                    "expected_keywords": relevant_keywords,
                    "found_keywords": found_keywords,
                    "quality_score": len(found_keywords) / len(relevant_keywords) if relevant_keywords else 0
                })
                
            except Exception as e:
                print(f"Error generating response for {case['id']}: {e}")
        
        results["avg_response_length"] = total_length / len(sample_cases) if sample_cases else 0
        results["responses_with_keywords"] = keyword_matches / len(sample_cases) if sample_cases else 0
        
        return results
    
    def _measure_performance(self, vdb: SimpleVectorDB) -> Dict[str, float]:
        """Measure query performance metrics."""
        # Sample query for timing
        sample_query = "What is Flash Attention?"
        
        # Time multiple queries
        times = []
        for _ in range(5):
            start_time = time.time()
            try:
                vdb.query(sample_query)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception:
                pass
        
        return {
            "avg_query_time": sum(times) / len(times) if times else 0.0,
            "min_query_time": min(times) if times else 0.0,
            "max_query_time": max(times) if times else 0.0
        }
    
    def compare_approaches(self, baseline_results: Dict, improved_results: Dict) -> Dict[str, Any]:
        """Compare baseline vs improved RAG approaches."""
        comparison = {
            "similarity_improvements": {},
            "retrieval_improvements": {},
            "overall_assessment": ""
        }
        
        # Compare similarity statistics
        baseline_sim = baseline_results["similarity_stats"]
        improved_sim = improved_results["similarity_stats"]
        
        comparison["similarity_improvements"] = {
            "cv_change": improved_sim["coefficient_of_variation"] - baseline_sim["coefficient_of_variation"],
            "mean_change": improved_sim["mean_similarity"] - baseline_sim["mean_similarity"],
            "std_change": improved_sim["std_similarity"] - baseline_sim["std_similarity"]
        }
        
        # Compare retrieval quality
        baseline_retrieval = baseline_results["retrieval_quality"]
        improved_retrieval = improved_results["retrieval_quality"]
        
        comparison["retrieval_improvements"] = {
            "hit_rate_change": improved_retrieval["hit_rate_top1"] - baseline_retrieval["hit_rate_top1"],
            "difficulty_improvements": {}
        }
        
        for difficulty in ["easy", "medium", "hard"]:
            baseline_rate = baseline_retrieval["by_difficulty"].get(difficulty, 0)
            improved_rate = improved_retrieval["by_difficulty"].get(difficulty, 0)
            comparison["retrieval_improvements"]["difficulty_improvements"][difficulty] = improved_rate - baseline_rate
        
        # Overall assessment
        cv_improved = comparison["similarity_improvements"]["cv_change"] > 0.01  # Higher CV is better
        hit_rate_improved = comparison["retrieval_improvements"]["hit_rate_change"] > 0.05  # 5% improvement
        
        if cv_improved and hit_rate_improved:
            comparison["overall_assessment"] = "SIGNIFICANT_IMPROVEMENT"
        elif cv_improved or hit_rate_improved:
            comparison["overall_assessment"] = "MODERATE_IMPROVEMENT"
        else:
            comparison["overall_assessment"] = "MINIMAL_IMPROVEMENT"
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of results."""
        print("\n" + "="*60)
        print("RAG EVALUATION RESULTS SUMMARY")
        print("="*60)
        
        # Similarity statistics
        sim_stats = results["similarity_stats"]
        print(f"\nSIMILARITY STATISTICS:")
        print(f"  Mean similarity: {sim_stats['mean_similarity']:.3f}")
        print(f"  Std deviation: {sim_stats['std_similarity']:.3f}")
        print(f"  Coefficient of variation: {sim_stats['coefficient_of_variation']:.3f}")
        
        # Retrieval quality
        retrieval = results["retrieval_quality"]
        print(f"\nRETRIEVAL QUALITY:")
        print(f"  Hit rate (top-1): {retrieval['hit_rate_top1']:.1%}")
        print(f"  By difficulty:")
        for difficulty, rate in retrieval["by_difficulty"].items():
            print(f"    {difficulty}: {rate:.1%}")
        
        # Performance
        perf = results["performance_metrics"]
        print(f"\nPERFORMANCE:")
        print(f"  Average query time: {perf['avg_query_time']:.3f}s")
        
        print("\n" + "="*60)


def main():
    """Run the evaluation pipeline."""
    print("Flash Attention RAG Evaluation Pipeline")
    print("="*50)
    
    # Initialize evaluator
    evaluator = FlashAttentionRAGEvaluator()
    
    # Print test dataset summary
    evaluator.evaluator.print_dataset_summary()
    
    # Check if VDB file exists
    import os
    if not os.path.exists(evaluator.vdb_file):
        print(f"\nError: Vector database file '{evaluator.vdb_file}' not found.")
        print("Please create the vector database first by running:")
        print("python create_vdb.py <path_to_flash_attention_paper.pdf>")
        return
    
    # Evaluate baseline
    baseline_results = evaluator.evaluate_baseline()
    
    # Print and save results
    evaluator.print_results_summary(baseline_results)
    evaluator.save_results(baseline_results, "baseline_evaluation.json")
    
    print(f"\nBaseline evaluation complete!")
    print(f"Now implement your improvements and run this script again to compare.")


if __name__ == "__main__":
    main()