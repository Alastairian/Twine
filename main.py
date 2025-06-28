import torch
from pathos_core import PathosMatrixOrchestrator
from lagos_marix import LogosMatrixOrchestrator

def run_twine_cognition(sensory_data):
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
    return final_plan

if __name__ == "__main__":
    # Example: Routine scenario
    routine_data = torch.rand(1, 10) * 0.2
    print("SCENARIO: Routine Situation")
    run_twine_cognition(routine_data)

    # Example: Urgent scenario
    danger_data = torch.rand(1, 10)
    danger_data[0, 3] = 0.95
    danger_data[0, 8] = 0.99
    print("\nSCENARIO: Urgent & Uncertain Situation")
    run_twine_cognition(danger_data)