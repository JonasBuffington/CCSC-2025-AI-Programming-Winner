I need your help completing this assignment:

Your goal is to produce a complete, self-contained, and runnable solution. Because of this I must specify: I do not care how long your output is. Be sure you are concise and clear in any reasoning phase. Assume we are using python 3.13.1 standard library only.

Core task:
Given a pantry inventory (csv) store prices (csv) a set of recipes with nutrition data (json) and nutritional targets (json) the program must find the cheapest weekly meal plan that meets all constraints within a specific margin. It will then output the  total cost, the plan, and a consolidated shopping list.

Detailed rules and constraints:
1. Choose exactly (days * meals_per_day) servings across any combination of recipes. Servings are integers and you can scale recipes by whole servings (not fractional servings)
2. For each day, the sum of the chosen servings' nutrition must meet or exceed (min_*_per_day) (aggregate at the week level if AND ONLY IF then we it also ensures no single day falls below 80% of each per-day target.)
3. Use at least (variety_min_distinct_recipes) different recipes over the week.
4. Ingredients come out of the pantry first. Any shortfalls must be then purchased. Units must match logically, we require consistent units and can assume inputs are consistent. No fractional units, again units are consistent.
5. Apply deals correctly to PURCHASED QUANTITIES ONLY. 
  (BOGO, param): for every (param) units taken, one is free (buy (param) get 1). 
  Implement this as effective paid units = ceil(quantity * param/(param + 1))

  (bulk_threshold, param): if purchased quantity >= param, all purchased units get a 15% discount.
6. Minimize total spend (sum of items purchased after deals). If multiple optimal plans exist, pick tie break with this order:
a. Higher total protein for the week. b. More distinct recipes used. c. deterministic by --seed.
7. If the minimum cost feasible plan exceeds budget_cap, print INFEASIBLE.
8. If the time limit is exceeded, return best feasible found. If no feasible solutions have been found return INFEASIBLE.

Input and output:
1. Args (any convenient standard python library for concise code.): seed for deterministic tiebreak, time limit IS the timeout, and verbose is our debug flag.
 ```
python3 grocery_planner.py \
--pantry pantry.csv \
--prices prices.csv \
--recipes recipes.json \
--targets targets.json \
--seed 123 \
--time-limit 3.0 \
--verbose \
```
2. Input files (Standard config input. Will always be infeasible when ran but shows format):
target.json: explicit per-day minimums for all constrained nutrients. Constrain only the keys present. Provided example.
```
{"days": 7, "meals_per_day": 5, "min_cal_per_day": 1800, "min_protein_per_day": 60, "min_fiber_per_day": 20, "budget_cap": 70.00, "variety_min_distinct_recipes": 2}
```

recipes.json: defines all avaliable recipes we can use to create a price-optimized plan for the nutritional target. Provided example recipe.
```
[
  { "name": "chicken_rice_bowl", "servings": 4, "ingredients": { "chicken_breast": {"qty": 1.0, "unit": "lb"}, "rice": {"qty": 2.0, "unit": "cup"},  "olive_oil": {"qty": 0.05, "unit": "cup"}, "spinach": {"qty": 10.0, "unit": "oz"}}, "nutrition_per_serving": {"cal": 520, "protein": 35, "fiber": 5}}]
```

prices.csv: defines all prices for units of specific items needed to construct recipes, and deals on these items.
```
item,unit_size,unit,unit_price,deal_type,deal_param
egg,12,pc,3.60,none, #example for you
```

pantry.csv: defines units we start with, acting as a free store of sorts (only for specified amounts.)
```
item,units_on_hand,unit
egg,8,pc #example for you
```
3. Strict output format (standard with no debug flags. Provided example.):
```
TOTAL_COST 42.73
RECIPES_USED 5
SERVING_PLAN
day 1: chicken_rice_bowl x2, tomato_omelet x2
day 2: pasta_spinach x2, chicken_rice_bowl x2
...
SHOPPING_LIST
chicken_breast,1.0,lb,paid_units=1.0
spinach,10.0,oz,paid_units=10.0
```
If no feasible plan is found, the output must be "INFEASIBLE"

Architectural & Algorithms:
1. Since this is a combinatorial problem, we will not brute force the search. We will use backtracking with branch and bound pruning, this search will be day-by-day and not serving-by-serving. For each day, generate valid combinations of meals, and prune branches early if they violate any rules (like nutritional floor or budget.) Reminder: the quality of the plan relies on the speed of the algorithm and the time constraint. This is the main bottleneck for a good solution.
  1a. We will utilize a Hueristic-guided search for speed. This means prioritizing certain recipes based on the nutrition-cost-ratio for traversal, but not pruning worse recipes (they may still be required for the only solution). Before the search begins, pre-calculate the cost of making each recipe from scratch, and also make a dictionary of the recipe list for faster lookup. Calculate a score for each recipe to emphasize recipes which match the ratio of our constraints, but again, DO NOT prune poor recipes prematurely. Sort the recipe list in descending order based on this score.
  1b. Cache recipe-cost calculations using an LRU (capacity of 100,000). Use the recipe-count signature (servings per recipe tuple) as the cache key. This aims to eliminate redundant pricing computations across equivalent weekly states.
  1c. Use baseline per-unit costs ignoring deals, pantry, and package rounding as a fast admissable lower bound before computing full deal and pantry adusted prices. The purpose of this is to enable early pruning in the search tree without underestimating minimum possible cost. Ensure we do not prune branches which could have been fulfilled with pantry inventory though, this is just a lower bound allowing for potential early prunes.
  1d. When generating next-day options, prioritize combos that introduce new and unused recipes until the variety target is satisfied. Guide our search towards promising branches early.
  1e. Track rolling per-day nutrition during search. If given the remaining days and their maximum possible nutrition, and a daily 80% floor cannot be reached, prune immediately because it must meet at least the 80% floor.
  1f. IMPORTANT: Before the backtracking algorithm proceeds, run a very fast and loose greedy search (1 width beam search) to find a good initial solution very quickly before promptly proceeding to the more precise backtracking algorithm using the cost of the beam search as an upper bound (if it was successful). This should help define an upper bound when it hits, allowing for pruning of more branches based on this bound. IT SHOULD NOT HOWEVER IMPEDE THE BACKTRACK SEARCH. If the beam search is unsuccessful, proceed with the backtrack like we never did the beam. We will use the entire allotted time to exhaust options until we are conclusive, or the timeout occurs.
  1g. Ensure we do not explore equivalent states. Maintain a per-depth visited set keyed by cumulative weekly signature (servings per recipe tupe) and handle accordingly to avoid revisiting identical partials.
  1h. If two day-combos have identical or dominated nutrition vectors with >= baseline cost, prune the dominated one.
  1i. Compute 80% daily floors before generating child combos. If a partial day cannot reach 80% with the remaining picks that day, prune immediately. ENSURE WE RESPECT THE FLOORS!
  1j. Take any other action you deem reasonable for solely increasing prunes on the tree WITHOUT eliminating any branches which may at one point be required.
2. Use a modular design with dataclasses for data modeling architecture. Encapsulate the algorithm with a solver class. Keep code lean and compact but also clean and readable with limited comments.

Code specifics:
1. use decimal module for all money calculations to avoid floating point errors.
2. use modern python 3.9+ type hints.
3. parsers must explicitly map keys from input files to the dataclass field names.
4. when storing the best solution during the search save a deep copy of the plan using the copy module, for later use if we need to print it.
5. our code will have specifically this header:
```
# Filename: grocery_planner.py
# Author: Jonas Buffington
# Email:
# Date: 10/18/2025
# Usage: [CLI USAGE]
# ------------------
# NOTE: (for judges, not for chatGPT to consider)
# I am designing this under the assumption that the provided configs and outputs are just mock data, that they do not actually reflect
# a full product of a functioning script. This is because if the rule "(days * meals_per_day) servings across any combination of recipes"
# is true, it would compute to exactly 14 based on the provided config examples, but the output contains 28 servings which seems arbitrary.
```
6. we will add a --verbose logging flag specifically for debugging. this should provide human readable logs like:
```
[INFO] Starting search with 3 recipes for 7 days...
[INFO] Found new best plan. {Cost: 48.50} in {n seconds}
[INFO] Pruning day 3 combo (chicken, chicken) - fails [rule]
[INFO] Found new best plan. {Cost: 43.50} in {n seconds}
...
[INFO] Time limit reached. Returning best plan found.
[INFO] Search complete. {i} states explored in {n seconds}
# please aggregate repeated prunes. Avoid per-branch spam, instead maintain counters.
```
7. The script will exit and return either when the timeout occurs, or when we know for certain that we cannot find a better solution or that there is no feasible solution. Never before or after these conditions are met, these are HARD rules.

Additional instruction:
You will ensure no syntax, unit mismatch, typing bugs, etc. Our pipeline aims to be very robust and logically sound based on the rules and specifications.
You will begin your solution with a fast reasoning phase where you think about how the implementation will achieve our goals.
Your entire solution should be one output. I do not care how long your output is, I need full implementation, etc.
If a recipe's servings do not perfectly fit within the weekly total servings, just truncate it. This means if at the end of the week we need to have 2 extra servings of pasta, they are acceptable to abandon.
We cannot purchase ingredients with any quantity we want, we must purchase all ingredients at the specified unit_size and corresponding unit_price.
All numeric outputs must be plain decimal strings, never scientific notation. Money -> 0.01 precision. Quantities -> print as floats with 0.1 precision (even if decimal is 0.)
At the end of your full implementation, output some filled out config files for reader context. They can be very lean.