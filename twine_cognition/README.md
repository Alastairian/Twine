# IAI-IPS Twine Cognition Prototype

This project is a prototype of the IAI-IPS Twine Cognition architecture for Artificial General Intelligence, featuring intertwined analytical ("Logos") and intuitive ("Pathos") cognitive cores.

## Features

- Simulated dual-core cognition: analytical and intuitive reasoning
- Bi-directional feedback between cores
- Demonstration scenarios: routine and urgent/uncertain situations
- CLI, Web API, logging, and unit tests

## Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/Alastairian/Twine.git
   cd Twine/twine_cognition
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

### CLI

Run the main prototype script:

```
python main.py --scenario routine
python main.py --scenario urgent
python main.py --scenario custom --custom_data 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

### Web API

Start the API server:

```
python app.py
```

Send a POST request (using curl, Postman, or similar):

```
curl -X POST http://127.0.0.1:5000/run -H "Content-Type: application/json" -d '{"sensory_data": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}'
```

### Logging

All runs are logged to `twine_cognition.log` for later review.

### Unit Tests

Run the core tests:

```
pytest test_core.py
```

## File Structure

```
twine_cognition/
├── main.py                # Main CLI runner
├── logger.py              # Logging utility
├── app.py                 # Web API (Flask)
├── pathos_core.py         # Pathos Core (Intuitive)
├── lagos_marix.py         # Logos Core (Analytical)
├── test_core.py           # Unit tests for cores
├── config.yaml            # Config file for parameters
├── requirements.txt       # Dependencies
├── README.md              # This file
```

## Custom Scenarios

To try your own input, use the CLI or Web API as shown above.

---

*For research and demonstration purposes only.*