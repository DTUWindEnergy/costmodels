import random
import sys
from pathlib import Path

import numpy as np

# --- Configuration ---
NUM_ITERATIONS = 100  # Number of random test cases to run
RTOL = 1e-9  # Relative tolerance for float comparison
ATOL = 1e-9  # Absolute tolerance for float comparison

# --- Add src directory to path to allow imports ---
# Assuming this script is in /home/ernie/code/costmodels/
script_dir = Path(__file__).parent.parent.parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# --- Import Models ---
try:
    from costmodels.models.dtu_offshore import (
        DTUOffshoreCostModel as OriginalDTUOffshoreCostModel,
    )
    from costmodels.models.dtu_offshore import Foundation as OriginalFoundation
    from costmodels.models.dtu_offshore import Quant as OriginalQuant

    print("Successfully imported Original DTUOffshoreCostModel.")
except ImportError as e:
    print(f"ERROR: Could not import original DTUOffshoreCostModel: {e}")
    print(
        "Make sure dtu_offshore.py exists in src/costmodels/models/ and is importable."
    )
    sys.exit(1)

try:
    # Assuming the refactored model is in dtu_offshore_concise.py
    from costmodels.dtu_offshore_concise import (
        DTUOffshoreCostModel as RefactoredDTUOffshoreCostModel,
    )
    from costmodels.dtu_offshore_concise import Foundation as RefactoredFoundation
    from costmodels.dtu_offshore_concise import Quant as RefactoredQuant

    print("Successfully imported Refactored DTUOffshoreCostModel.")
except ImportError as e:
    print(f"ERROR: Could not import refactored DTUOffshoreCostModel: {e}")
    print(
        "Make sure dtu_offshore_concise.py exists in src/costmodels/ and is importable."
    )
    sys.exit(1)


# --- Input Parameter Sampling ---
def sample_inputs():
    """Generates a dictionary of randomized inputs for the cost models."""
    nwt = random.randint(5, 50)
    lifetime = random.randint(20, 30)
    # Use the Foundation enum from one of the models (assuming they are compatible)
    foundation_option = random.choice(list(OriginalFoundation))

    # Use OriginalQuant for generating inputs initially
    inputs = {
        "rated_power": OriginalQuant(random.uniform(3.0, 15.0), "MW"),
        "rotor_speed": OriginalQuant(random.uniform(5.0, 15.0), "rpm"),
        "rotor_diameter": OriginalQuant(random.uniform(100.0, 240.0), "m"),
        "hub_height": OriginalQuant(random.uniform(90.0, 150.0), "m"),
        "foundation_option": foundation_option,
        "water_depth": OriginalQuant(random.uniform(10.0, 50.0), "m"),
        # "currency": "EURO/KW", # Keep fixed for simplicity, adjust if needed
        "eur_to_dkk": random.uniform(7.4, 7.6),
        "wacc": OriginalQuant(random.uniform(5.0, 10.0), "%"),  # Use percentage value
        "devex": OriginalQuant(random.uniform(800, 1200), "EUR/kW"),
        "decline_factor": OriginalQuant(
            random.uniform(0.1, 1.0), "%"
        ),  # Use percentage value
        "inflation": OriginalQuant(
            random.uniform(1.0, 3.0), "%"
        ),  # Use percentage value
        "lifetime": lifetime,
        "opex": OriginalQuant(random.uniform(15, 35), "EUR/kW"),
        "abex": OriginalQuant(0, "EUR"),  # Keep abex simple or add randomization
        "profit": OriginalQuant(
            random.uniform(10.0, 25.0), "%"
        ),  # Use percentage value
        # Provide capacity_factor, let internal logic calculate AEP if needed
        "capacity_factor": OriginalQuant(
            random.uniform(35.0, 55.0), "%"
        ),  # Use percentage value
        "aep": OriginalQuant(
            np.nan, "MWh"
        ),  # Set AEP to NaN when using capacity factor
        "nwt": nwt,
        "electrical_cost": OriginalQuant(random.uniform(1.0, 2.0), "MEUR/MW"),
        "eprice": OriginalQuant(
            random.uniform(0.05, 0.15), "EUR/kWh"
        ),  # Include if needed by models
    }
    return inputs


# --- Comparison Function ---
def compare_results(original_res, refactored_res, iteration):
    """Compares two result dictionaries, checking for close numerical values."""
    all_match = True
    mismatched_keys = []

    if not isinstance(original_res, dict) or not isinstance(refactored_res, dict):
        print(f"Iteration {iteration}: ERROR - Results are not dictionaries.")
        return False

    original_keys = set(original_res.keys())
    refactored_keys = set(refactored_res.keys())

    if original_keys != refactored_keys:
        print(f"Iteration {iteration}: WARNING - Result dictionary keys differ.")
        print(f"  Keys only in Original: {original_keys - refactored_keys}")
        print(f"  Keys only in Refactored: {refactored_keys - original_keys}")
        # Decide if this constitutes a failure based on requirements
        # all_match = False

    common_keys = original_keys.intersection(refactored_keys)

    for key in common_keys:
        val_orig = original_res[key]
        val_refac = refactored_res[key]

        # Extract magnitude if Quant object, otherwise use the value directly
        mag_orig = (
            val_orig.m
            if isinstance(val_orig, (OriginalQuant, RefactoredQuant))
            else val_orig
        )
        mag_refac = (
            val_refac.m
            if isinstance(val_refac, (OriginalQuant, RefactoredQuant))
            else val_refac
        )

        # Check if values are numeric (or numpy array)
        is_numeric_orig = isinstance(mag_orig, (int, float, np.number, np.ndarray))
        is_numeric_refac = isinstance(mag_refac, (int, float, np.number, np.ndarray))

        if is_numeric_orig and is_numeric_refac:
            try:
                # Handle potential NaN values - treat NaN == NaN as True
                # Handle potential Inf values - treat Inf == Inf as True
                match = np.allclose(
                    mag_orig, mag_refac, rtol=RTOL, atol=ATOL, equal_nan=True
                )

                # Explicit check for Inf mismatch if allclose returns True for Inf==Inf
                is_orig_inf = np.isinf(mag_orig)
                is_refac_inf = np.isinf(mag_refac)
                if np.any(
                    is_orig_inf != is_refac_inf
                ):  # If one has Inf where the other doesn't
                    match = False

                if not match:
                    print(f"Iteration {iteration}: MISMATCH for key '{key}':")
                    print(f"  Original:   {val_orig}")
                    print(f"  Refactored: {val_refac}")
                    diff = np.abs(
                        np.nan_to_num(mag_orig) - np.nan_to_num(mag_refac)
                    )  # Calculate diff ignoring NaNs
                    print(f"  Difference: {diff}")
                    mismatched_keys.append(key)
                    all_match = False
            except TypeError as e:
                # Handle cases where np.allclose fails (e.g., incompatible array shapes)
                print(f"Iteration {iteration}: TYPE ERROR comparing key '{key}': {e}")
                print(f"  Original:   {val_orig} (type: {type(mag_orig)})")
                print(f"  Refactored: {val_refac} (type: {type(mag_refac)})")
                mismatched_keys.append(key)
                all_match = False
            except Exception as e:
                print(
                    f"Iteration {iteration}: UNEXPECTED ERROR comparing key '{key}': {e}"
                )
                mismatched_keys.append(key)
                all_match = False
        elif mag_orig != mag_refac:  # Non-numeric comparison
            print(f"Iteration {iteration}: MISMATCH (non-numeric) for key '{key}':")
            print(f"  Original:   {val_orig} (type: {type(mag_orig)})")
            print(f"  Refactored: {val_refac} (type: {type(mag_refac)})")
            mismatched_keys.append(key)
            all_match = False

    if not all_match:
        print(f"Iteration {iteration}: FAILED. Mismatched keys: {mismatched_keys}")
    # else:
    #     print(f"Iteration {iteration}: PASSED.") # Uncomment for verbose success logging

    return all_match


# --- Main Monte Carlo Loop ---
failures = 0
print(f"\n--- Starting Monte Carlo Simulation ({NUM_ITERATIONS} iterations) ---")

for i in range(NUM_ITERATIONS):
    print(f"\n--- Iteration {i+1}/{NUM_ITERATIONS} ---")
    inputs = sample_inputs()

    # Prepare inputs for each model (handle potential Quant/Enum differences if necessary)
    # If Quant/Foundation are identical, these steps might be simplified
    inputs_orig = inputs
    inputs_refac = {
        k: (RefactoredQuant(v.m, v.u) if isinstance(v, OriginalQuant) else v)
        for k, v in inputs.items()
    }
    try:
        # Ensure Foundation enum values are compatible
        inputs_refac["foundation_option"] = RefactoredFoundation(
            inputs_orig["foundation_option"].value
        )
    except ValueError as e:
        print(
            f"Iteration {i+1}: ERROR - Incompatible Foundation enum value: {inputs_orig['foundation_option']}"
        )
        failures += 1
        continue

    org_result = None
    refac_result = None

    # Run Original Model
    try:
        # print(f"  Inputs (Original): {inputs_orig}") # Debug input
        org_model = OriginalDTUOffshoreCostModel(**inputs_orig)
        org_result = org_model._run()
        # print(f"  Result (Original): {org_result}") # Debug result
    except Exception as e:
        print(f"Iteration {i+1}: ERROR running original model: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        failures += 1
        # Continue to next iteration even if one model fails, to test the other
        # Or 'continue' here if comparison requires both results

    # Run Refactored Model
    try:
        # print(f"  Inputs (Refactored): {inputs_refac}") # Debug input
        refac_model = RefactoredDTUOffshoreCostModel(**inputs_refac)
        refac_result = refac_model._run()
        # print(f"  Result (Refactored): {refac_result}") # Debug result
    except Exception as e:
        print(f"Iteration {i+1}: ERROR running refactored model: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for detailed traceback
        failures += 1
        # Continue to next iteration

    # Compare results if both runs were successful
    if org_result is not None and refac_result is not None:
        print("  Comparing results...")
        if not compare_results(org_result, refac_result, i + 1):
            # Increment failure count only if comparison fails
            # (Errors during run are already counted)
            # Check if failure already counted for this iteration due to run error
            is_run_error = org_result is None or refac_result is None
            if not is_run_error:  # Avoid double counting if run error occurred
                failures += 1  # This logic might need refinement based on desired failure counting
    elif org_result is None or refac_result is None:
        print("  Skipping comparison due to error during model execution.")


# --- Final Report ---
print("\n--- Monte Carlo Test Summary ---")
print(f"Total iterations: {NUM_ITERATIONS}")
print(f"Number of failed iterations (mismatches or execution errors): {failures}")
if failures == 0:
    print("SUCCESS: All test iterations passed!")
else:
    print(f"FAILURE: {failures} iterations failed. Check log for details.")
    sys.exit(1)  # Exit with error code if tests fail

sys.exit(0)
