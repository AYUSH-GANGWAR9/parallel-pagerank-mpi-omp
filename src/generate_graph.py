#!/usr/bin/env python3
"""
Generate a random directed graph edge list (u v per line).
python3 src/generate_graph.py --n 10000 --m 50000 --out data/edges.txt
"""
import argparse, random, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--m', type=int, default=5000)
    p.add_argument('--out', type=str, default='data/edges.txt')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        written = 0
        while written < args.m:
            u = random.randrange(0, args.n)
            v = random.randrange(0, args.n)
            if u == v: continue
            f.write(f"{u} {v}\n")
            written += 1
    print(f"Wrote {args.m} edges to {args.out}")

if __name__ == '__main__':
    main()
