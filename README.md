# Save Jessica - Morty Express Challenge

This project helps you interact with the [Sphinx Morty Express Challenge](https://challenge.sphinxhq.com/).

## Challenge Overview

Your goal is to send 1000 Morties from the Citadel to Planet Jessica through one of three intermediate planets:

- **Planet A (index=0)**: "On a Cob" Planet
- **Planet B (index=1)**: Cronenberg World
- **Planet C (index=2)**: The Purge Planet

The risk for each planet changes dynamically based on the number of trips taken. Your objective is to maximize the number of Morties who arrive safely!

### The Twist

The survival probability for each planet **changes over time** based on the number of trips taken. This is the key to the challenge!

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your API Token

1. Visit <https://challenge.sphinxhq.com/>
2. Request a token with your name and email
3. Create .env file

```bash
echo "SPHINX_API_TOKEN=your_token_here" > .env
```

### 3. Run the Example

```bash
python example.py
```

### Console Output

You'll see survival rates for each planet:

```text
"On a Cob" Planet:
  Survival Rate: 67.50%
  
Cronenberg World:
  Survival Rate: 68.20%
  
The Purge Planet:
  Survival Rate: 65.80%
```

### Visualizations

Several plots will appear showing:

1. **Survival rates over time** - How each planet performs as trips increase
2. **Overall comparison** - Bar chart comparing planets
3. **Moving average** - Trends in survival rates
4. **Risk evolution** - How risk changes (early vs late trips)
5. **Episode summary** - Complete dashboard

## Project Structure

- `api_client.py` - API client with all endpoint functions
- `data_collector.py` - Functions to collect and analyze data
- `visualizations.py` - Functions to visualize challenge data
- `example.py` - Example usage script
- `strategy.py` - Template for building your own strategy

## API Functions

The `SphinxAPIClient` class provides:

- `start_episode()` - Initialize a new escape attempt
- `send_morties(planet, morty_count)` - Send Morties through a portal
- `get_status()` - Check current progress

## Next Steps

### Option 1: Implement a Strategy

Edit `strategy.py` to create your own strategy:

```python
from strategy import run_strategy, SimpleGreedyStrategy

# Run a pre-built strategy
run_strategy(SimpleGreedyStrategy, explore_trips=30)

# Or create your own by subclassing MortyRescueStrategy
```

### Option 2: Custom Script

Create your own Python script:

```python
from api_client import SphinxAPIClient
from data_collector import DataCollector

# Initialize
client = SphinxAPIClient()
collector = DataCollector(client)

# Start episode
client.start_episode()

# Your strategy here...
```

## Checking Your Progress

At any time, check your status:

```python
status = client.get_status()
print(f"Saved: {status['morties_on_planet_jessica']}")
print(f"Lost: {status['morties_lost']}")
print(f"Remaining: {status['morties_in_citadel']}")
```

## API Not Working?

```python
# Test your connection
from api_client import SphinxAPIClient

client = SphinxAPIClient()
status = client.get_status()
print(status)
```