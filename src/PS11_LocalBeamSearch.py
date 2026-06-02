import sys
import traceback

def read_input(file_path):
    """
    Reads the input grid file.
    Expected format:
    Line 1: rows cols
    Line 2: start_r start_c
    Line 3: goal_r goal_c
    Line 4 to end: grid rows (space separated)
    """
    print(f"Reading input file from path: {file_path}\n")
    
    with open(file_path, 'r') as f:
        raw_lines = f.readlines()
    
    # Show exactly what is in the file (including empty lines)
    print("=== RAW CONTENT OF INPUT FILE ===")
    for i, line in enumerate(raw_lines):
        print(f"Line {i}: '{line.strip()}'")
    print("================================\n")
    
    # Clean lines (remove empty)
    lines = [line.strip() for line in raw_lines if line.strip()] # Remove empty lines
    
    if len(lines) < 8:
        raise ValueError("Input file does not have enough lines!") # We expect at least 3 lines for metadata + 5 lines for grid (for a 5x5 grid)
    
    # Parse
    rows, cols = map(int, lines[0].split())  # First line: rows and columns
    start_row, start_col = map(int, lines[1].split())  # Second line: start position
    goal_row, goal_col = map(int, lines[2].split())  # Third line: goal position
    
    print("=== INPUT METADATA ===")
    print(f"Rows          : {rows}")
    print(f"Columns       : {cols}")
    print(f"Start Position: ({start_row}, {start_col})")
    print(f"Goal Position : ({goal_row}, {goal_col})")
    print("======================\n")
    
    # Read grid
    grid = []
    for i in range(3, 3 + rows):
        row = lines[i].split()
        grid.append(row)
    
    return rows, cols, (start_row, start_col), (goal_row, goal_col), grid 


def get_cost(cell):
    return 3 if cell == 'C' else 1


def is_valid(r, c, rows, cols, grid):
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'


def manhattan(r, c, goal):
    return abs(r - goal[0]) + abs(c - goal[1])


def local_beam_search(rows, cols, start, goal, grid, k=2):
    print("\n=== Local Beam Search Execution (k=2) ===\n")
    current_beam = [(start, 0, [start])]
    iteration = 0

    while current_beam:
        print(f"Iteration {iteration}:")
        for (r, c), g, _ in current_beam:
            h = manhattan(r, c, goal)
            print(f"  Position: ({r},{c})  |  h = {h}  |  g = {g}")
        
        for (r, c), g, path in current_beam:
            if (r, c) == goal:
                print("\n*** GOAL REACHED ***")
                return path, g
        
        candidates = []
        for (r, c), g, path in current_beam:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc, rows, cols, grid):
                    cost = get_cost(grid[nr][nc])
                    new_g = g + cost
                    new_path = path + [(nr, nc)]
                    candidates.append(((nr, nc), new_g, new_path))
        
        if not candidates:
            print("No path found!")
            return None, None
        
        best = {}
        for pos, g, path in candidates:
            if pos not in best or g < best[pos][1]:
                best[pos] = (pos, g, path)
        
        candidate_list = list(best.values())
        candidate_list.sort(key=lambda x: (manhattan(x[0][0], x[0][1], goal), x[1]))
        
        current_beam = candidate_list[:k]
        iteration += 1
    
    return None, None


# ====================== MAIN ======================
if __name__ == "__main__":
    try:
        input_file = "D:\\Ree_Learning\\SEM_2\\ACI\\ACI_Assignment1\\input\\inputPS11.txt"
        output_file = "D:\\Ree_Learning\\SEM_2\\ACI\\ACI_Assignment1\\output\\outputPS11.txt"

        print("=" * 70)
        print("FILE PATHS USED:")
        print(f"Input File Path  →  {input_file}")
        print(f"Output File Path →  {output_file}")
        print("=" * 70 + "\n")

        rows, cols, start, goal, grid = read_input(input_file)

        print(f"Grid Size: {rows} x {cols}\n")

        path, total_cost = local_beam_search(rows, cols, start, goal, grid, k=2)

        if path:
            path_str = " -> ".join([f"({r},{c})" for r, c in path])
            print("\n" + "="*70)
            print("FINAL RESULT")
            print("="*70)
            print(f"Final Path : {path_str}")
            print(f"Total Cost : {total_cost}")
            print("="*70)

            with open(output_file, "w") as f:
                f.write(f"Final Path from Start to Goal:\n{path_str}\n")
                f.write(f"Total Path Cost: {total_cost}\n")
                f.write(f"Total Optimized Traversal Cost: {total_cost}\n")
            print(f"\nOutput written to: {output_file}")
        else:
            print("No solution found.")

    except Exception as e:
        print("\n❌ ERROR OCCURRED!")
        print(f"Error: {e}")
        traceback.print_exc()