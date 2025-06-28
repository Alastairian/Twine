import torch
import argparse
from pathos_core import PathosMatrixOrchestrator
from lagos_marix import LogosMatrixOrchestrator
from logger import log_decision

def run_twine_cognition(sensory_data, scenario):
    print("\n=== Twine Cognition Prototype ===\n")
    logos = LogosMatrixOrchestrator()
    pathos = PathosMatrixOrchestrator(sensory_input_size=sensory_data.shape[1])

    # 1. Pathos generates "gut feeling"
    intuitive_markers = pathos.run_intuitive_analysis(sensory_data)

    # 2. Logos performs logical analysis
    _ = logos.run_logical_analysis(sensory_data)

    # 3. Logos's plan is adjusted by Pathos's strategic command
    final_plan = logos.adjust_plan_with_pathos_guidance(intuitive_markers)
    print("\n--- Final Decision ---")
    print(final_plan)

    # Log the run
    log_decision(scenario, sensory_data, final_plan)
    return final_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twine Cognition CLI")
    parser.add_argument("--scenario", choices=["routine", "urgent", "custom"], default="routine")
    parser.add_argument("--custom_data", nargs=10, type=float, help="10 float values for custom sensory input")
    args = parser.parse_args()

    if args.scenario == "routine":
        sensory_data = torch.rand(1, 10) * 0.2
        print("SCENARIO: Routine Situation")
    elif args.scenario == "urgent":
        sensory_data = torch.rand(1, 10)
        sensory_data[0, 3] = 0.95
        sensory_data[0, 8] = 0.99
        print("SCENARIO: Urgent & Uncertain Situation")
    else:
        if args.custom_data is None:
            print("Please provide --custom_data with 10 float values.")
            exit(1)
        sensory_data = torch.tensor([args.custom_data])
        print("SCENARIO: Custom")

    run_twine_cognition(sensory_data, args.scenario)