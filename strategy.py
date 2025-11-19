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
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
import os

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


class StatsWithAllPlanetsOnlyProbsEval(MortyRescueStrategy):

    def execute_strategy(self, eta=0.2, last_results_size_C=8, last_results_size_B=6, last_results_size_A=4):

        client = self.client
        status = client.get_status()

        morties_remaining = status["morties_in_citadel"]

        planet_names = ["Planet A", "Planet B", "Planet C"]
        #Période des différentes planètes:
        periods = {0: 10, 1: 20, 2: 200}

        #Liste des résultats récents pour chaque planète:
        recent = {0: [], 1: [], 2: []}

        #Taille des tableaux des récents résultats:
        window = {0: last_results_size_A, 1: last_results_size_B, 2: last_results_size_C}

        #Booléen informant si une planète a été initialisée:
        initialized = {0: False, 1: False, 2: False}

        #Pour chaque planète, un dictionnaire contenant pour chaque step la probabilité de survivre
        good_cycles = {0: {}, 1: {}, 2: {}}

        #Définis si une planète est sur une winstreak
        winstreak = {0: False, 1: False, 2: False}

        #Définis si une fin de cycle doit être détectée
        close_window = {0: False, 1: False, 2: False}

        total_steps = 0

        best_pid = None
        best_prob = (0, None)


        #Petits paramètres d'optimisation lors de la détection du cycle de la planète C
        count_streak_C = 0
        send_3 = True

        #Chargement des probas de survie
        prob_A = np.array([0.1, 0.2, 0.4, 0.7, 1., 1., 0.7, 0.4, 0.2, 0.1])
        if os.path.exists("prob_A.npy"):
            prob_A = np.load("prob_A.npy")

        prob_B = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1., 1., 0.9, 0.7, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
        if os.path.exists("prob_B.npy"):
            prob_B = np.load("prob_B.npy")

        prob_C = np.array([1.]*55 + [0.9]*10 + [0.7]*10 + [0.5]*10 + [0.3]*10 + [0.1]*105)
        if os.path.exists("prob_C.npy"):
            prob_C = np.load("prob_C.npy")

        #Mettre a jour le tableau des récents résultats
        def update_recent(pid, survived):
            recent[pid].append(survived)
            if len(recent[pid]) > window[pid]:
                recent[pid].pop(0)

        #Détecte la fin de cycle
        def window_closed(pid):
            """The planet is now in its bad period."""
            print(f"Recents values: {recent[pid]}")
            if len(recent[pid]) < window[pid]:
                return False
            if pid == 2:
                return sum(recent[pid]) <= max(window[pid] // 2, 2)
            else:
                return sum(recent[pid]) <= window[pid]-1

        #Définis les probabilités pour chaque planète
        def record_good_period(pid, end_step):
            period = periods[pid]
            if pid == 0:
                middle = end_step - 5
                while middle <= 1010:
                    good_cycles[pid][middle] =  (prob_A[4], 4)
                    good_cycles[pid][middle + 1] = (prob_A[5], 5)
                    good_cycles[pid][middle + 2] = (prob_A[6], 6)
                    good_cycles[pid][middle - 1] = (prob_A[3], 3)
                    good_cycles[pid][middle + 3] = (prob_A[7], 7)
                    good_cycles[pid][middle - 2] = (prob_A[2], 2)
                    good_cycles[pid][middle + 4] = (prob_A[8], 8)
                    good_cycles[pid][middle - 3] = (prob_A[1], 1)
                    good_cycles[pid][middle + 5] = (prob_A[9], 9)
                    good_cycles[pid][middle - 4] = (prob_A[0], 0)
                    middle += 10
            elif pid == 1:
                middle = end_step - 6
                while middle <= 1020:
                    good_cycles[pid][middle] = (prob_B[9], 9)
                    good_cycles[pid][middle + 1] = (prob_B[10], 10)
                    good_cycles[pid][middle - 1] = (prob_B[8], 8)
                    good_cycles[pid][middle + 2] = (prob_B[11], 11)
                    good_cycles[pid][middle - 2] = (prob_B[7], 7)
                    good_cycles[pid][middle + 3] = (prob_B[12], 12)
                    good_cycles[pid][middle - 3] = (prob_B[6], 6)
                    good_cycles[pid][middle + 4] = (prob_B[13], 13)
                    good_cycles[pid][middle - 4] = (prob_B[5], 5)
                    good_cycles[pid][middle + 5] = (prob_B[14], 14)
                    good_cycles[pid][middle - 5] = (prob_B[4], 4)
                    good_cycles[pid][middle + 6] = (prob_B[15], 15)
                    good_cycles[pid][middle - 6] = (prob_B[3], 3)
                    good_cycles[pid][middle + 7] = (prob_B[16], 16)
                    good_cycles[pid][middle - 7] = (prob_B[2], 2)
                    good_cycles[pid][middle + 8] = (prob_B[17], 17)
                    good_cycles[pid][middle - 8] = (prob_B[1], 1)
                    good_cycles[pid][middle + 9] = (prob_B[18], 18)
                    good_cycles[pid][middle - 9] = (prob_B[0], 0)
                    good_cycles[pid][middle + 10] = (prob_B[19], 19)
                    middle += 20
            else:
                middle = end_step - period//4 + 10
                while middle <= 1200:
                    good_cycles[pid][middle] =(min(prob_C[0] + 0.05, 1.), 0)
                    for i in range(1, 99):
                        j=2*i
                        if i <= 5:
                            val1 = min(prob_C[j] + 0.05, 1.)
                            val2 = min(prob_C[j+1] + 0.05, 1.)
                        else:
                            val1 = prob_C[j]
                            val2 = prob_C[j+1]
                        good_cycles[pid][middle + i] = (val1, j)
                        good_cycles[pid][middle - i] = (val2, j+1)
                    good_cycles[pid][middle - 100] = (prob_C[199], 199)
                    middle += 200
            initialized[pid] = True
            close_window[pid] = False
            winstreak[pid] = False

        while morties_remaining > 0:
            
            ### CHOIX PLANETE

            best_pid = None
            best_prob = (-1, None)
            
            #Détection de la meilleure planète en terme de probabilités
            for pid in (0, 1, 2):
                prob = good_cycles[pid].get(total_steps, (0.0, None))  # par défaut 0 si pas de valeur
                if prob[0] > best_prob[0]:
                    best_prob = prob
                    best_pid = pid

            #Si personne n'est en winstreak, qu'aucun cycle ne doit être fermé, et que la meilleure proba est plus de 0.8, on prend la planète associée
            if not any(winstreak.values()) and not any(close_window.values()) and best_prob[0] >= 0.8:
                best_pid = best_pid
                best_prob = best_prob
                
            else:
                #Si toutes les planètes n'ont pas été initialisées
                if not all(initialized.values()):
                    best_prob = (0, None)
                    #On vérifie d'abord si un cycle doit être fermé. Le cas échéant, on choisit la planète associée.
                    if close_window[2]:
                        count_streak_C += 1
                        best_pid = 2
                    elif close_window[1]:
                        best_pid = 1
                    elif close_window[0]:
                        best_pid = 0
                    #Si une planète est en winstreak, on continue dessus jusqu'à détecter un cycle.
                    elif winstreak[2]:
                        best_pid = 2
                    elif winstreak[1]:
                        best_pid = 1
                    elif winstreak[0]:
                        best_pid = 0
                    else:
                        #On prend aléatoirement une planète non initialisée qu'on teste
                        candidates = [pid for pid in (2,1,0) if not initialized[pid]]
                        if len(candidates) == 1:
                            best_pid = candidates[0]
                        elif len(candidates) == 2:
                            if total_steps % 2 == 0:
                                best_pid = candidates[0]
                            else:
                                best_pid = candidates[1]
                        else:
                            if total_steps % 2 == 0:
                                best_pid = 2
                            else:
                                if total_steps % 4 == 1:
                                    best_pid = 0
                                else:
                                    best_pid = 1
                else:
                    #Si toutes les planètes ont été initialisées, on prend la meilleure proba
                    best_prob = best_prob
                    best_pid = best_pid

            ### CHOIX NOMBRE MORTIES A ENVOYER
            #Si on est sur A ou B et qu'on est sur winstreak, on envoie 3 (optimisation sur un bon épisode)
            if winstreak[best_pid] and not close_window[best_pid] and not best_pid == 2:
                send = min(3, morties_remaining)
            #Si on cherche la fin de cycle de C, et qu'on a pas encore croisé de 0, on envoie 3, sinon, 1.
            elif not initialized[best_pid] and best_pid == 2 and count_streak_C < 40:
                if send_3:
                    send = min(3, morties_remaining)
                else:
                    send = 1
            #Sur la planete A, on envoie jamais 3. Sur les autres, on envoie si la proba vaut plus de 0.8
            elif best_prob[0] > 0.8 and best_pid != 0:
                send = min(3, morties_remaining)
            #On envoie 2 si la proba est plus de 0.7
            elif best_prob[0] > 0.7:
                send = min(2, morties_remaining)
            else:
                send = 1
            print(f"Step {total_steps+1}: Sending {send} to {planet_names[best_pid]} with prob {best_prob[0]}")
            result = client.send_morties(best_pid, send)

            survived = int(result["survived"])

            print(f"Survived: {survived}")
            if not survived and best_pid == 2 and close_window[2]:
                send_3 = False

            #Initialisation winstreak
            if survived and not initialized[best_pid]:
                winstreak[best_pid] = True
            else:
                winstreak[best_pid] = False

            morties_remaining = result["morties_in_citadel"]

            update_recent(best_pid, survived)
            
            if initialized[best_pid]:
                ### MISE A JOUR PROBABILITES
                eta = eta
                P_min = 0.05

                reward = survived

                # Période de la planète
                period = periods[best_pid]

                # Step actuel utilisé pour l'apprentissage
                base_step = total_steps

                # On met à jour tous les steps futurs équivalents dans le cycle
                # (step + k*period)
                step_to_update = base_step
                while step_to_update <= 1000:   # limite arbitraire, ajustable
                    old_p = good_cycles[best_pid].get(step_to_update, (0.0, None))

                    # Policy gradient simple : P ← P + η * (reward − P)
                    advantage = reward - old_p[0]
                    new_p = (old_p[0] + eta * advantage, old_p[1])

                    new_val = max(P_min, min(1.0, new_p[0]))
                    new_p = (new_val, new_p[1])

                    good_cycles[best_pid][step_to_update] = new_p

                    if best_pid == 0:
                        prob_A[old_p[1]] = new_p[0]
                    elif best_pid == 1:
                        prob_B[old_p[1]] = new_p[0]
                    else:
                        prob_C[old_p[1]] = new_p[0]

                    step_to_update += period

            total_steps += 1

            #Si on est sur une bonne winstreak, on commence à chercher la fin du cycle
            if sum(recent[best_pid]) > window[best_pid] - 1 and not initialized[best_pid]:
                close_window[best_pid] = True


            # Détection de la fin de la première bonne période
            if not initialized[best_pid] and close_window[best_pid] and window_closed(best_pid):
                record_good_period(best_pid, total_steps)

        # ----------------------------------------
        # FINAL
        # ----------------------------------------
        #Sauvegarde des probabilités
        if len(prob_A) != 0:
            np.save("prob_A.npy", prob_A)
        if len(prob_B) != 0:
            np.save("prob_B.npy",prob_B)
        if len(prob_C) != 0:
            np.save("prob_C.npy",prob_C)
        final = client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(final)

        return (final["morties_on_planet_jessica"] / 1000) * 100

def run_strategy(strategy_class):
    client = SphinxAPIClient()
    client.start_episode()
    strategy = strategy_class(client)
    rate = strategy.execute_strategy(eta = 0.2, last_results_size_C=8, last_results_size_B=6, last_results_size_A=4)
    print(f"Rate: {rate:.4f}")

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
    
    run_strategy(StatsWithAllPlanetsOnlyProbsEval)