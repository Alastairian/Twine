import logging

logging.basicConfig(
    filename="twine_cognition.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

def log_decision(scenario, sensory_data, decision):
    logging.info("SCENARIO: %s | INPUT: %s | DECISION: %s", scenario, sensory_data.tolist(), decision)