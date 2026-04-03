#!/usr/bin/env python3
"""
Estimate the number of valid spanning subgraphs for d-dimensional hypercube.

This helps determine appropriate sampling parameters for training.

Key insights:
1. Total search space: 2^(d * 2^(d-1)) subgraphs
2. But we only care about SPANNING subgraphs (all vertices)
3. With diameter constraint = d (not just connected)
4. The actual valid count is MUCH smaller than theoretical max

Usage:
    python estimate_valid_subgraphs.py --dimension 5
"""

import argparse
from math import comb, log2
from itertools import combinations
from collections import deque
import numpy as np


def build_hypercube_adjacency(d):
    """Build adjacency list for d-dimensional hypercube."""
    n_vertices = 2 ** d
    adj = {i: [] for i in range(n_vertices)}
    
    for u in range(n_vertices):
        for bit in range(d):
            v = u ^ (1 << bit)  # Flip bit to get neighbor
            if u < v:  # Avoid duplicates
                adj[u].append(v)
                adj[v].append(u)
    
    return adj


def get_all_hypercube_edges(d):
    """Get all edges in d-dimensional hypercube."""
    n_vertices = 2 ** d
    edges = []
    
    for u in range(n_vertices):
        for bit in range(d):
            v = u ^ (1 << bit)
            if u < v:
                edges.append((u, v))
    
    return edges


def compute_diameter(n_vertices, edge_list):
    """
    Compute diameter of graph with given edges.
    Returns infinity if disconnected or diameter > expected.
    """
    # Build adjacency list
    adj = {i: [] for i in range(n_vertices)}
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check connectivity with BFS from vertex 0
    visited = set()
    queue = deque([0])
    visited.add(0)
    
    while queue:
        current = queue.popleft()
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    if len(visited) < n_vertices:
        return float('inf')  # Disconnected
    
    # Compute all-pairs shortest paths
    max_dist = 0
    for start in range(n_vertices):
        # BFS from start
        dist = {start: 0}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    max_dist = max(max_dist, dist[neighbor])
                    queue.append(neighbor)
        
        if len(dist) < n_vertices:
            return float('inf')  # Shouldn't reach here if connected
    
    return max_dist


def monte_carlo_estimate(d, num_samples=100000, min_edges=8, max_edges=16):
    """
    Estimate number of valid subgraphs using Monte Carlo sampling.
    
    Strategy:
    1. Sample subgraphs with k edges (for different k values)
    2. Check what fraction have diameter = d
    3. Extrapolate to total count
    """
    print(f"\n{'='*70}")
    print(f"Monte Carlo Estimation for d={d} Hypercube")
    print(f"{'='*70}\n")
    
    n_vertices = 2 ** d
    all_edges = get_all_hypercube_edges(d)
    total_edges = len(all_edges)
    
    print(f"Basic parameters:")
    print(f"  Dimension d: {d}")
    print(f"  Vertices: {n_vertices}")
    print(f"  Total edges in Q_{d}: {total_edges}")
    print(f"  Search space size: 2^{total_edges} ≈ {2**total_edges:.2e}")
    print(f"\n")
    
    results = {}
    
    for k in range(min_edges, max_edges + 1):
        if k > total_edges:
            break
        
        total_combinations = comb(total_edges, k)
        valid_count = 0
        
        print(f"Testing k={k} edges (C({total_edges},{k}) = {total_combinations:.2e} combinations)...")
        
        # Sample random subgraphs with exactly k edges
        for _ in range(num_samples):
            # Randomly select k edges
            sampled_edges = [all_edges[i] for i in np.random.choice(
                total_edges, size=k, replace=False
            )]
            
            # Check diameter
            diam = compute_diameter(n_vertices, sampled_edges)
            if diam == d:
                valid_count += 1
        
        validity_rate = valid_count / num_samples
        estimated_valid = int(total_combinations * validity_rate)
        
        results[k] = {
            'total': total_combinations,
            'valid_in_sample': valid_count,
            'validity_rate': validity_rate,
            'estimated_total_valid': estimated_valid
        }
        
        print(f"  Valid in sample: {valid_count}/{num_samples} ({validity_rate*100:.2f}%)")
        if estimated_valid > 0:
            print(f"  Estimated total valid: ~{estimated_valid:,}")
            print(f"  Log10 of estimates: {log2(estimated_valid)/log2(10):.1f} (i.e., 10^{log2(estimated_valid)/log2(10):.1f})\n")
        else:
            print(f"  Estimated total valid: < 1 (extremely rare!)\n")

    # Aggregate estimates
    print(f"\n{'='*70}")
    print("SUMMARY: Estimated Valid Subgraphs by Edge Count")
    print(f"{'='*70}")
    
    total_estimated = sum(r['estimated_total_valid'] for r in results.values())
    
    for k, data in sorted(results.items()):
        bar_len = int(data['estimated_total_valid'] / max(1, total_estimated) * 50)
        bar = "█" * bar_len
        print(f"k={k:2d} edges: {data['estimated_total_valid']:>12,} valid  {bar}")
    
    print(f"\nTotal estimated valid subgraphs: ~{total_estimated:,}")
    print(f"Log10: {log2(total_estimated)/log2(10):.1f} (i.e., 10^{log2(total_estimated)/log2(10):.1f})")
    
    # Calculate entropy and diversity
    print(f"\n{'='*70}")
    print("IMPLICATIONS FOR SAMPLING STRATEGY")
    print(f"{'='*70}")
    
    if total_estimated > 0:
        # Coupon collector estimate
        n_solutions = total_estimated
        coupon_collector_need = int(n_solutions * np.log(n_solutions))
        
        print(f"\nCoupon Collector Analysis:")
        print(f"  To collect ALL {n_solutions:,} solutions:")
        print(f"    Need approximately: {coupon_collector_need:,} samples")
        print(f"    (Formula: N × ln(N) where N = number of solutions)")
        
        print(f"\nPractical Sampling Recommendations:")
        
        # Realistic coverage goals
        for coverage in [0.01, 0.05, 0.10, 0.50]:
            samples_needed = int(-n_solutions * np.log(1 - coverage))
            print(f"  For {coverage*100:5.1f}% coverage: ~{samples_needed:,} samples needed")
        
        print(f"\nTraining Configuration Advice:")
        
        if n_solutions < 1e6:
            print(f"  ✓ Small search space (< 1M solutions)")
            print(f"  → Recommended per epoch: 50k-100k samples")
            print(f"  → Recommended epochs: 50-100")
            print(f"  → Total samples: 5M-10M should be SUFFICIENT")
        elif n_solutions < 1e9:
            print(f"  ⚠ Medium search space (1M-1B solutions)")
            print(f"  → Recommended per epoch: 100k-500k samples")
            print(f"  → Recommended epochs: 200-500")
            print(f"  → Total samples: 20M-100M recommended")
        else:
            print(f"  ❌ Large search space (> 1B solutions)")
            print(f"  → Recommended per epoch: 500k-1M samples")
            print(f"  → Recommended epochs: 500-1000")
            print(f"  → Total samples: 250M+ may still be insufficient")
            print(f"  → Consider smarter sampling strategies!")
    
    print(f"\n{'='*70}\n")
    
    return results


def analytical_lower_bound(d):
    """
    Calculate known constructions (Graham-Harary) as lower bound.
    
    Graham-Harary construction gives us a specific solution with:
    - For d=4: 8 edges
    - For d=5: approximately 12-14 edges (exact value varies by construction)
    """
    print(f"\nAnalytical Lower Bounds (Known Constructions):")
    print(f"{'='*50}")
    
    # Graham-Harary construction edge count
    gh_edges = 2 * d + 2 * (d - 1)
    
    print(f"Graham-Harary construction for d={d}:")
    print(f"  Approximate edges: {gh_edges}")
    print(f"  This provides at least ONE valid solution")
    print(f"  But there are likely MANY more with similar edge counts\n")
    
    return gh_edges


def main():
    parser = argparse.ArgumentParser(description='Estimate valid hypercube subgraphs')
    parser.add_argument('--dimension', type=int, default=5, help='Hypercube dimension')
    parser.add_argument('--samples', type=int, default=50000, help='Monte Carlo samples per edge count')
    parser.add_argument('--min-edges', type=int, default=8, help='Minimum edges to test')
    parser.add_argument('--max-edges', type=int, default=16, help='Maximum edges to test')
    
    args = parser.parse_args()
    
    if args.dimension > 6:
        print(f"WARNING: d={args.dimension} will be very slow!")
        print(f"Consider d<=5 for practical estimation.\n")
    
    # Analytical bounds
    analytical_lower_bound(args.dimension)
    
    # Monte Carlo estimation
    monte_carlo_estimate(
        args.dimension,
        num_samples=args.samples,
        min_edges=args.min_edges,
        max_edges=args.max_edges
    )


if __name__ == "__main__":
    main()
