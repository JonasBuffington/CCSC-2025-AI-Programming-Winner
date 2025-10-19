# AI-Generated Grocery Optimizer (1st Place Winner)

**1st place, CCSC 2025 Student Programming Competition**

This repository contains my winning submission for the 2025 CCSC Student Programming Competition. The challenge was to create a complex grocery cost and nutrition optimizer in Python withn a 24-hour period.

The primary constraint was interesting: **the entire application had to be generated from a single, final ChatGPT prompt.** No manual coding or bug-fixing on the final script was allowed, and I could not give ChatGPT any Python code.

This project represents my learning in one-shotting in AI-assisted development. While AI-assisted development is not foreign to me (at all), I mostly work iteratively. This project required a different kind of iteration than I am used to.

## The Challenge

The task was to build a Python script that could ingest multiple data sources (CSV, JSON) to create and solve a complex combinatorial optimization problem. The goal was to find the absolute cheapest meal plan that met strict constraints in: length (days), nutrition values, variety, and budget.

A naive brute-force approach would not be scalable as the search space grows extremely fast in longer or more complex config environments.

## The Single Prompt Constraint

The competition rules forbade manual coding for submissions. The entire 24-hour development process was an exercise in debugging and iteration *at the prompt level.* I had to:
1. Architect the entire program logic in natural language.
2. Specify all data structures, algorithms, and program architecture.
3. Debug incorrect outputs by refining my prompt's logic. Not by fixing the code.
4. Iteratively engineer the prompt to instruct the AI to build the advanced optimization and pruning approach.

You can view the final, winning prompt that generated `grocery_planner.py` here:
**[View the Winning Prompt](./WINNING_PROMPT.md)**

## Performance Results

My prompt successfully guided the AI to build a highly efficient combinatiorial searching algorithm. On the provided stress-test data suite, using my own machine, the final program:
* Found the optimal solution in **0.001 seconds**
* Proved optimality by exploring the entire valid search space in **0.481 seconds**
* Intelligently **pruned over 200,000** invalid states to efficiently solve the problem.

## How to Run

1. Clone the repository
2. Run the script from the root directory, pointing it to the data files:

```bash
python3 grocery_planner.py \
  --pantry ./data/pantry.csv \
  --prices ./data/prices.csv \
  --recipes ./data/recipes.json \
  --targets ./data/targets.json \
  --time-limit 3.0 \
  [--verbose] \
  ```
