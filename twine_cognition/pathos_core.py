import torch
import torch.nn as nn
import numpy as np

# --- ====================================================== ---
# --- PHASE 3 ARTIFACT: Simulating the Two Cores             ---
# --- ====================================================== ---

# A simple neural network to represent the core of a Pathos Cognitive Unit.
# Its job is to learn abstract patterns from raw data.
class PathosUnit(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()
        # In a real system, this would be a complex CNN/Transformer front-end.
        # Here, we simulate it with a simple MLP.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        # This head predicts the intuitive markers.
        self.marker_head = nn.Sequential(
            nn.Linear(10, output_size),
            nn.Sigmoid() # Output values between 0 and 1
        )
        
    def forward(self, raw_sensory_data):
        features = self.feature_extractor(raw_sensory_data)
        intuitive_markers = self.marker_head(features)
        # The output corresponds to [urgency, salience, uncertainty]
        return intuitive_markers

class PathosMatrixOrchestrator:
    """
    The Intuitive Core. It "feels" a situation and provides guiding markers.
    """
    def __init__(self, sensory_input_size=10):
        # In a real system, this unit would be highly trained via RL.
        # Here, we initialize it as a new "brain."
        self.core_unit = PathosUnit(input_size=sensory_input_size)
        print("Pathos Core initialized.")

    def run_intuitive_analysis(self, raw_sensory_data):
        print("Pathos: Analyzing raw sensory data for intuitive feel...")
        with torch.no_grad():
            markers_tensor = self.core_unit(raw_sensory_data)
        
        markers = {
            "urgency": markers_tensor[0][0].item(),
            "salience": markers_tensor[0][1].item(),
            "uncertainty": markers_tensor[0][2].item()
        }
        print(f"Pathos: Intuitive markers generated -> {markers}")
        return markers

class LogosMatrixOrchestrator:
    """
    The Logical Core. It performs analysis and formulates plans.
    (Simplified for this simulation)
    """
    def __init__(self):
        self.current_plan = None
        print("Logos Core initialized.")

    def run_logical_analysis(self, raw_sensory_data):
        print("Logos: Performing initial logical analysis...")
        # In a real system, this would involve the 4 cognitive modes.
        # Here, we simulate a simple outcome.
        if raw_sensory_data.mean() > 0.5:
            self.current_plan = {"action": "Proceed with Optimized Route", "confidence": 0.95}
        else:
            self.current_plan = {"action": "Proceed with Standard Route", "confidence": 0.98}
        print(f"Logos: Preliminary plan formulated -> {self.current_plan}")
        return self.current_plan

    def adjust_plan_with_pathos_guidance(self, intuitive_markers):
        """
        Applies the strategic command from Pathos to its own plan.
        """
        print("Logos: Receiving strategic command from Pathos...")
        # This is the "Twine" connection in action.
        # The logic here implements the "Strategic Command Interface."
        if intuitive_markers["urgency"] > 0.8 and intuitive_markers["uncertainty"] > 0.6:
            print("Logos: High urgency and uncertainty detected! Overriding plan.")
            self.current_plan = {"action": "HALT. Re-evaluate. Run Deeper Analysis on high-salience targets.", "confidence": 0.99}
        elif intuitive_markers["urgency"] > 0.8:
            print("Logos: High urgency detected! Switching to conservative plan.")
            self.current_plan["action"] = "Switch to Conservative Failsafe Route"
        
        print(f"Logos: Final, adjusted plan -> {self.current_plan}")
        return self.current_plan

# --- The Integrated Twine Cognition System ---

class TwineCognitionSystem:
    def __init__(self, sensory_input_size=10):
        print("--- Twine Cognition System Initializing ---")
        self.logos = LogosMatrixOrchestrator()
        self.pathos = PathosMatrixOrchestrator(sensory_input_size=sensory_input_size)
        print("--- System Ready ---")

    def run_cognitive_cycle(self, raw_sensory_data):
        print(f"\n--- New Cognitive Cycle Initiated ---")
        print(f"Raw Sensory Input (mean value): {raw_sensory_data.mean():.2f}")
        
        # 1. Pathos generates its "gut feeling" first.
        intuitive_markers = self.pathos.run_intuitive_analysis(raw_sensory_data)
        
        # 2. Logos performs its analysis in parallel.
        _ = self.logos.run_logical_analysis(raw_sensory_data)
        
        # 3. Logos's plan is *forced* to be adjusted by Pathos's strategic command.
        final_plan = self.logos.adjust_plan_with_pathos_guidance(intuitive_markers)
        
        print("--- Cognitive Cycle Complete ---")
        return final_plan

# --- Main Execution: Simulating the System in Action ---
if __name__ == "__main__":
    # Initialize the full mind
    iai_ips_mind = TwineCognitionSystem(sensory_input_size=10)
    
    # --- SCENARIO 1: A "Routine" Situation ---
    # Low-variance, predictable sensory data.
    routine_data = torch.rand(1, 10) * 0.2 
    final_plan_1 = iai_ips_mind.run_cognitive_cycle(routine_data)
    print(f"\nFINAL DECISION (Routine): {final_plan_1['action']}\n" + "="*40)
    
    # --- SCENARIO 2: An "Urgent & Uncertain" Situation ---
    # High-variance, unpredictable data with certain "danger" spikes.
    # This is what Pathos should be trained to "feel" as dangerous.
    danger_data = torch.rand(1, 10)
    danger_data[0, 3] = 0.95 # spike
    danger_data[0, 8] = 0.99 # bigger spike
    
    final_plan_2 = iai_ips_mind.run_cognitive_cycle(danger_data)
    print(f"\nFINAL DECISION (Urgent): {final_plan_2['action']}\n" + "="*40)