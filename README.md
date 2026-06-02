# ACI Assignment 1 - PS11: Route Optimizer

**Local Beam Search (k=2) Implementation**

**Course**: MTech in Artificial Intelligence and Machine Learning  
**Assignment**: Route Optimizer Agent using Local Beam Search

---

## Problem Description

An intelligent agent must find the **optimal low-cost route** from Start (`S`) to Goal (`G`) in a 5x5 warehouse grid.  
- Moves allowed: **Up, Down, Left, Right** (No diagonals)  
- Normal cells cost = **1**  
- High-cost cells (`C`) cost = **3** (1 + 2 penalty)  
- Blocked cells (`X`) cannot be traversed  
- Heuristic used: **Manhattan Distance**
