import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- ====================================================== ---
# --- PHASE 1 ARTIFACT: The Trainable Cognitive Unit (Upgraded) ---
# --- ====================================================== ---

class IAI_IPS_Unit(nn.Module):
    def __init__(self, input_size=2, l4_nodes=5, l5_nodes=5):
        super().__init__()
        self.input_size = input_size
        self.layer2 = Layer2_OpposingAspects() if input_size == 2 else nn.Identity()
        self.layer3 = Layer3_Contradictions() if input_size == 2 else nn.Identity()
        
        l4_input_dim = 5 if input_size == 2 else input_size
        
        self.l4 = nn.Linear(l4_input_dim, l4_nodes)
        self.l5 = nn.Linear(l4_nodes, l5_nodes)
        self.core_node_gate = nn.Sequential(nn.Linear(3 if input_size == 2 else l4_input_dim, 10), nn.ReLU(), nn.Linear(10, l5_nodes), nn.Sigmoid())
        self.l6 = nn.Linear(l5_nodes, 5)
        self.l7 = nn.Linear(5, 4)
        self.l8 = nn.Linear(4, 3)
        self.l9 = nn.Linear(3, 2)
        self.l10 = nn.Linear(2, 1) 
        self.activation = nn.Sigmoid()
        
        # --- Multi-purpose heads for different learning modes ---
        self.next_state_head = nn.Sequential(nn.Linear(2, input_size), nn.Tanh()) # For Predictive
        self.previous_state_head = nn.Sequential(nn.Linear(2, input_size), nn.Tanh()) # For Historical
        self.abstraction_head = nn.Sequential(nn.Linear(2, 2)) # For Deeper

    def forward(self, x):
        # Initial IAI-IPS layers for surface-level analysis
        if self.input_size == 2:
            l2_out = self.layer2(x)
            l3_out = self.layer3(l2_out)
            l_2_3_combined = torch.cat([l2_out, l3_out], dim=1)
            gate_input = l3_out
            l4_input = l_2_3_combined
        else: # For deeper/abstract inputs
            l4_input = x
            gate_input = x
        
        # Main processing path
        l4_out = self.activation(self.l4(l4_input))
        l5_out = self.activation(self.l5(l4_out))
        gate = self.core_node_gate(gate_input)
        adjusted_l5_output = l5_out * gate
        l6_out = self.activation(self.l6(adjusted_l5_output))
        l7_out = self.activation(self.l7(l6_out))
        l8_out = self.activation(self.l8(l7_out))
        l9_out = self.activation(self.l9(l8_out)) # Final internal representation
        
        # --- Generate outputs for all potential modes ---
        predicted_outcome = self.activation(self.l10(l9_out))
        predicted_next_state = self.next_state_head(l9_out)
        predicted_previous_state = self.previous_state_head(l9_out)
        output_abstraction = self.abstraction_head(l9_out)

        # We also need the contradiction vector for Generalized learning
        contradiction_vector = l3_out if self.input_size == 2 else None
        
        return predicted_outcome, output_abstraction, predicted_next_state, predicted_previous_state, contradiction_vector

class Layer2_OpposingAspects(nn.Module): # Unchanged
    def __init__(self):
        super().__init__()
        self.V_raw = nn.Parameter(torch.tensor([0.1, 0.1]))
        self.V_ripe = nn.Parameter(torch.tensor([0.9, 0.9]))
        self.gamma = nn.Parameter(torch.tensor(2.0))
    def forward(self, x):
        gamma = torch.relu(self.gamma) + 1e-6
        dist_to_raw = torch.norm(x - self.V_raw, dim=-1, keepdim=True)
        dist_to_ripe = torch.norm(x - self.V_ripe, dim=-1, keepdim=True)
        a_raw = torch.exp(-gamma * (dist_to_raw**2)); a_ripe = torch.exp(-gamma * (dist_to_ripe**2))
        return torch.cat([a_raw, a_ripe], dim=1)

class Layer3_Contradictions(nn.Module): # Unchanged
    def __init__(self): super().__init__()
    def forward(self, l2_output):
        a_raw, a_ripe = l2_output[:, 0].unsqueeze(1), l2_output[:, 1].unsqueeze(1)
        a_int = 1 - torch.abs(a_raw - a_ripe); a_ext = 1 - a_ripe; a_intext = a_int * a_ext
        return torch.cat([a_int, a_ext, a_intext], dim=1)


# --- ====================================================== ---
# --- PHASE 2, SPRINT 3: The Full Cognitive Matrix           ---
# --- ====================================================== ---

class LogosMatrixOrchestrator:
    def __init__(self, pretrained_unit_path=None):
        print("Logos Matrix Orchestrator initialized. All four modes available.")
        self.pretrained_unit_path = pretrained_unit_path
        # --- Simulated Knowledge Base for Generalized Learning ---
        self.knowledge_base = {
            "Physics": torch.tensor([0.1, 0.9, 0.09]), # Low internal, high external contradiction
            "Biology": torch.tensor([0.8, 0.5, 0.40]), # High internal, medium external
            "Finance": torch.tensor([0.5, 0.8, 0.40]), # Medium internal, high external
            "Philosophy": torch.tensor([0.9, 0.1, 0.09]) # High internal, low external
        }

    def _spawn_unit(self, input_size): # Unchanged
        unit = IAI_IPS_Unit(input_size=input_size)
        if self.pretrained_unit_path and input_size == 2: unit.load_state_dict(torch.load(self.pretrained_unit_path))
        unit.eval(); return unit

    def run_deeper_analysis(self, initial_data, depth=2): # Unchanged
        print(f"\n--- Running Deeper Analysis (Depth: {depth}) ---")
        analysis_chain = []
        current_input = initial_data
        for i in range(depth):
            unit = self._spawn_unit(input_size=current_input.shape[1])
            with torch.no_grad(): _, next_input, _, _, _ = unit(current_input)
            print(f"Layer {i+1}: Input shape {current_input.shape} -> Output abstraction shape {next_input.shape}")
            analysis_chain.append({"abstraction": next_input})
            current_input = next_input
        return analysis_chain

    def run_predictive_analysis(self, current_state, potential_actions): # Unchanged
        print(f"\n--- Running Predictive Analysis for {len(potential_actions)} actions ---")
        predictions = {}
        for i, action in enumerate(potential_actions):
            state_action_input = torch.cat([current_state, action], dim=1)
            unit = self._spawn_unit(input_size=state_action_input.shape[1])
            with torch.no_grad(): outcome, next_state, _, _, _ = unit(state_action_input)
            predictions[f"Action_{i+1}"] = {"outcome": outcome.item(), "next_state": next_state.numpy()}
        return predictions

    def run_historical_analysis(self, current_state, steps_back=2):
        print(f"\n--- Running Historical Analysis (Steps Back: {steps_back}) ---")
        history = []
        state_to_analyze = current_state
        for i in range(steps_back):
            unit = self._spawn_unit(input_size=state_to_analyze.shape[1])
            with torch.no_grad(): _, _, _, prev_state, _ = unit(state_to_analyze)
            print(f"Step -{i+1}: Analyzing state shape {state_to_analyze.shape} -> Inferred previous state shape {prev_state.shape}")
            history.append({"inferred_past_state": prev_state.numpy()})
            state_to_analyze = prev_state
        return history

    def run_generalized_analysis(self, initial_problem_data):
        print(f"\n--- Running Generalized Analysis ---")
        unit = self._spawn_unit(input_size=initial_problem_data.shape[1])
        with torch.no_grad(): _, _, _, _, contradiction_vector = unit(initial_problem_data)
        
        if contradiction_vector is None:
            print("Cannot run generalized analysis on abstract data without a contradiction vector.")
            return None

        print(f"Problem's Contradiction Fingerprint: {contradiction_vector.numpy().flatten()}")
        
        similarities = {}
        for domain, domain_vector in self.knowledge_base.items():
            # Using Cosine Similarity to find the most similar conceptual domain
            similarity = F.cosine_similarity(contradiction_vector, domain_vector.unsqueeze(0))
            similarities[domain] = similarity.item()
        
        most_analogous_domain = max(similarities, key=similarities.get)
        print(f"Analogical search complete. Highest similarity with domain: '{most_analogous_domain}'")
        return most_analogous_domain, similarities

# --- Main Execution ---
if __name__ == "__main__":
    torch.manual_seed(42) # for reproducible results
    # --- PHASE 1: Prepare Base Unit ---
    torch.save(IAI_IPS_Unit(input_size=2).state_dict(), 'base_unit.pth')

    # --- PHASE 2: Full Orchestrator Demonstration ---
    print("--- Phase 2: Initializing the Logos Matrix with all four cognitive modes ---")
    orchestrator = LogosMatrixOrchestrator(pretrained_unit_path='base_unit.pth')
    
    # Define a surface-level problem
    initial_problem = torch.tensor([[0.6, 0.7]]) # A somewhat ripe fruit
    
    # --- Execute All Four Modes ---
    
    # 1. GENERALIZED: What kind of problem is this?
    domain, _ = orchestrator.run_generalized_analysis(initial_problem)
    
    # 2. DEEPER: What are the underlying components of this problem?
    deep_analysis = orchestrator.run_deeper_analysis(initial_problem)
    most_abstract_state = deep_analysis[-1]["abstraction"]
    
    # 3. HISTORICAL: How did we get to this abstract state?
    history = orchestrator.run_historical_analysis(most_abstract_state)
    
    # 4. PREDICTIVE: From this abstract state, what happens if we act?
    actions = [torch.tensor([[0.1, 0.1]]), torch.tensor([[0.9, -0.9]])]
    predictions = orchestrator.run_predictive_analysis(most_abstract_state, actions)
    
    print("\n--- COGNITIVE MATRIX SUMMARY ---")
    print(f"Initial Problem: {initial_problem.numpy().flatten()}")
    print(f"1. Generalized Analysis: Problem is most analogous to '{domain}'.")
    print(f"2. Deeper Analysis: Processed problem into an abstract state of shape {most_abstract_state.shape}.")
    print(f"3. Historical Analysis: Inferred {len(history)} past states leading to this abstraction.")
    print(f"4. Predictive Analysis: Forecasted outcomes for {len(predictions)} potential actions.")
    print("\n--- Phase 2 (Logos Matrix) is feature-complete. ---")