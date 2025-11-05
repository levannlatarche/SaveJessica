"""
Strategy template for the Morty Express Challenge.

This file provides a template for implementing your own strategy
to maximize the number of Morties saved.

The challenge: The survival probability of each planet changes over time
(based on the number of trips taken). Your strategy should adapt to these
changing conditions.
"""

from api_client import SphinxAPIClient
from data_collector import DataCollector
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

class MortyRescueStrategy:
    """Base class for implementing rescue strategies."""
    
    def __init__(self, client: SphinxAPIClient):
        """
        Initialize the strategy.
        
        Args:
            client: SphinxAPIClient instance
        """
        self.client = client
        self.collector = DataCollector(client)
        self.exploration_data = []
    
    def explore_phase(self, trips_per_planet: int = 30) -> pd.DataFrame:
        """
        Initial exploration phase to understand planet behaviors.
        
        Args:
            trips_per_planet: Number of trips to send to each planet
            
        Returns:
            DataFrame with exploration data
        """
        print("\n=== EXPLORATION PHASE ===")
        df = self.collector.explore_all_planets(
            trips_per_planet=trips_per_planet,
            morty_count=1  # Send 1 Morty at a time during exploration
        )
        self.exploration_data = df
        return df
    
    def analyze_planets(self) -> dict:
        """
        Analyze planet data to determine characteristics.
        
        Returns:
            Dictionary with analysis results
        """
        if len(self.exploration_data) == 0:
            raise ValueError("No exploration data available. Run explore_phase() first.")
        
        return self.collector.analyze_risk_changes(self.exploration_data)
    
    def execute_strategy(self):
        """
        Execute the main rescue strategy.
        Override this method to implement your own strategy.
        """
        raise NotImplementedError("Implement your strategy in a subclass")


class SimpleGreedyStrategy(MortyRescueStrategy):
    """
    Simple greedy strategy: always pick the planet with highest recent success.
    """
    
    def execute_strategy(self, morties_per_trip: int = 3):
        """
        Execute the greedy strategy.
        
        Args:
            morties_per_trip: Number of Morties to send per trip (1-3)
        """
        print("\n=== EXECUTING GREEDY STRATEGY ===")
        
        # Get current status
        status = self.client.get_status()
        morties_remaining = status['morties_in_citadel']
        
        print(f"Starting with {morties_remaining} Morties in Citadel")
        
        # Determine best planet from exploration
        best_planet, best_planet_name = self.collector.get_best_planet(
            self.exploration_data,
            consider_trend=True
        )
        
        print(f"Best planet identified: {best_planet_name}")
        print(f"Sending all remaining Morties to {best_planet_name}...")
        
        trips_made = 0
        
        while morties_remaining > 0:
            # Determine how many to send
            morties_to_send = min(morties_per_trip, morties_remaining)
            
            # Send Morties
            result = self.client.send_morties(best_planet, morties_to_send)
            
            morties_remaining = result['morties_in_citadel']
            trips_made += 1
            
            if trips_made % 50 == 0:
                print(f"  Progress: {trips_made} trips, "
                      f"{result['morties_on_planet_jessica']} saved, "
                      f"{morties_remaining} remaining")
        
        # Final status
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")


class IdiotStrategy(MortyRescueStrategy):
    def execute_strategy(self, batch_size: int = 3, window_size: int = 30):
        """
        Stratégie "idiote" améliorée :
        - Envoie 3 Mortys sur chaque planète au départ.
        - Puis envoie toujours sur la planète avec le meilleur taux de survie
          calculé sur les 30 derniers Mortys envoyés sur chaque planète.
        """
        print("\n=== EXECUTING SMARTER IDIOT STRATEGY ===")

        client = self.client
        try:
            status = client.get_status()
            morties_remaining = status['morties_in_citadel']
            print(f"✓ Connected to API — {morties_remaining} Mortys available.")
        except Exception as e:
            print(f"✗ Error initializing API: {e}")
            return

        collector = DataCollector(client)
        planet_ids = [0, 1, 2]
        planet_names = ["Planet A", "Planet B", "Planet C"]

        # Journal des envois
        all_data = pd.DataFrame(columns=['planet', 'sent', 'survived'])

        print("\n=== INITIAL EXPLORATION ===")
        for pid in planet_ids:
            if morties_remaining <= 0:
                break
            morties_to_send = min(batch_size, morties_remaining)
            result = client.send_morties(pid, morties_to_send)
            survived = result['survived']

            all_data.loc[len(all_data)] = [pid, morties_to_send, survived]
            morties_remaining = result['morties_in_citadel']

            print(f"  → Sent {morties_to_send} to {planet_names[pid]} | Survived: {survived}")

        round_idx = 1
        while morties_remaining > 0:
            print(f"\n=== ROUND {round_idx} ===")

            survival_rates = {}
            for pid in planet_ids:
                # Sélection des derniers envois pour cette planète
                recent_rows = all_data[all_data['planet'] == pid].tail(window_size // batch_size)
                if len(recent_rows) == 0:
                    survival_rates[pid] = 0
                    continue

                sent_recent = recent_rows['sent'].sum()
                survived_recent = recent_rows['survived'].sum()
                survival_rates[pid] = survived_recent / sent_recent if sent_recent > 0 else 0

            # Affichage des taux récents
            for pid, rate in survival_rates.items():
                print(f"  {planet_names[pid]} → recent survival rate: {rate*100:.2f}%")

            # Choisir la meilleure planète sur la base des 30 derniers Mortys
            best_planet = max(survival_rates, key=survival_rates.get)
            best_rate = survival_rates[best_planet]
            print(f"\n  🚀 Sending next batch to {planet_names[best_planet]} ({best_rate*100:.2f}%)")

            morties_to_send = min(batch_size, morties_remaining)
            result = client.send_morties(best_planet, morties_to_send)
            survived = result['survived']
            morties_remaining = result['morties_in_citadel']

            all_data.loc[len(all_data)] = [best_planet, morties_to_send, survived]

            round_idx += 1

        # Fin de partie
        final_status = client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

        all_data.to_csv("idiot_strategy_log.csv", index=False)
        print("\n✓ Saved log to idiot_strategy_log.csv")

# ===== Réseau DQN =====
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ===== Mémoire de replay =====
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayMemory:
    def __init__(self, capacity=5000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAllocationStrategy(MortyRescueStrategy):
    def __init__(self, client: SphinxAPIClient):
        super().__init__(client)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DQN hyperparams
        self.state_dim = 3    # (x, y, z)
        self.action_dim = 3   # choix de planète
        self.gamma = 0.95
        self.lr = 1e-3
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_send = 9  # nombre de mortys envoyés à chaque action

        # Réseaux
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def execute_strategy(self, sync_target_every=10):
        print("\n=== EXECUTING DQN ALLOCATION STRATEGY ===")

        client = self.client
        status = client.get_status()
        morties_remaining = status['morties_in_citadel']
        collector = DataCollector(client)

        planet_names = ["Planet A", "Planet B", "Planet C"]
        state = np.array([0, 0, 0], dtype=np.float32)
        total_sent = 0
        step = 0
        history = []

        while total_sent < 1000:
            # Normaliser l’état
            state_norm = state / 1000.0

            # Choix d'action (planète)
            action = self.select_action(state_norm)
            planet = action

            # Envoyer les mortys
            morties_to_send = min(self.batch_send, 1000 - total_sent)
            result1 = client.send_morties(int(planet), int(min(morties_to_send, 3)))
            new_morties_to_send = morties_to_send - min(morties_to_send, 3)
            if new_morties_to_send <= 0:
                survived = result1['survived']
            else:
                result2 = client.send_morties(int(planet), int(min(new_morties_to_send, 3)))
                new_morties_to_send -= min(new_morties_to_send, 3)
                if new_morties_to_send <= 0:
                    survived = result1['survived'] + result2['survived']
                else:
                    result3 = client.send_morties(int(planet), int(min(new_morties_to_send, 3)))
                    survived = result1['survived'] + result2['survived'] + result3['survived']

            # Calcul de la récompense
            reward = survived / morties_to_send
            done = (total_sent + morties_to_send) >= 1000

            # État suivant
            next_state = state.copy()
            next_state[planet] += morties_to_send

            # Stocker transition
            self.memory.push(state_norm, action, reward, next_state / 1000.0, done)
            self.optimize_model()

            # Mettre à jour
            state = next_state
            total_sent += morties_to_send
            step += 1

            if step % sync_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            print(f"Step {step}: Sent {morties_to_send} to {planet_names[planet]} | "
                  f"Reward={reward:.2f} | State={state.tolist()} | ε={self.epsilon:.3f}")

            history.append({
                'step': step,
                'planet': planet,
                'sent': morties_to_send,
                'survived': survived,
                'reward': reward,
                'epsilon': self.epsilon
            })

        pd.DataFrame(history).to_csv("dqn_allocation_log.csv", index=False)
        print("\n✓ Saved training log to dqn_allocation_log.csv")

def run_strategy(strategy_class, explore_trips: int = 30):
    """
    Run a complete strategy from exploration to execution.
    
    Args:
        strategy_class: Strategy class to use
        explore_trips: Number of exploration trips per planet
    """
    # Initialize client and strategy
    client = SphinxAPIClient()
    strategy = strategy_class(client)
    
    # Start new episode
    print("Starting new episode...")
    client.start_episode()
    
    # Exploration phase
    strategy.explore_phase(trips_per_planet=explore_trips)
    
    # Analyze results
    analysis = strategy.analyze_planets()
    print("\nPlanet Analysis:")
    for planet_name, data in analysis.items():
        print(f"  {planet_name}: {data['overall_survival_rate']:.2f}% "
              f"({data['trend']})")
    
    # Execute strategy
    strategy.execute_strategy()


if __name__ == "__main__":
    print("Morty Express Challenge - Strategy Module")
    print("="*60)
    
    print("\nAvailable strategies:")
    print("1. SimpleGreedyStrategy - Pick best planet and stick with it")
    print("2. AdaptiveStrategy - Monitor and adapt to changing conditions")
    
    print("\nExample usage:")
    print("  run_strategy(SimpleGreedyStrategy, explore_trips=30)")
    print("  run_strategy(AdaptiveStrategy, explore_trips=30)")
    
    print("\nTo create your own strategy:")
    print("1. Subclass MortyRescueStrategy")
    print("2. Implement the execute_strategy() method")
    print("3. Use self.client to interact with the API")
    print("4. Use self.collector to analyze data")
    
    # Uncomment to run:
    run_strategy(DQNAllocationStrategy, explore_trips=10)
