"""
API Client for Sphinx Morty Express Challenge
Base URL: https://challenge.sphinxhq.com
"""

import requests
import os
from typing import Dict, Optional
from dotenv import load_dotenv


class SphinxAPIClient:
    """Client for interacting with the Sphinx Morty Express Challenge API."""
    
    BASE_URL = "https://challenge.sphinxhq.com"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_token: API token for authentication. If not provided, 
                      will try to load from SPHINX_API_TOKEN environment variable.
        """
        load_dotenv()
        self.api_token = api_token or os.getenv("SPHINX_API_TOKEN")
        
        if not self.api_token:
            raise ValueError(
                "API token is required. Either pass it as an argument or "
                "set SPHINX_API_TOKEN environment variable."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def request_token(self, name: str, email: str) -> Dict:
        """
        Request an API token (one-time setup).
        Token will be sent to the provided email.
        
        Args:
            name: Your name
            email: Your email address
            
        Returns:
            Response from the API
        """
        url = f"{self.BASE_URL}/api/auth/request-token/"
        payload = {
            "name": name,
            "email": email
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def start_episode(self) -> Dict:
        """
        Start a new episode. This initializes your escape attempt.
        
        Returns:
            Dict with keys:
                - morties_in_citadel: Number of Morties still in the Citadel
                - morties_on_planet_jessica: Number of Morties on Planet Jessica
                - morties_lost: Number of Morties lost
                - steps_taken: Number of trips taken
                - status_message: Status message
        """
        url = f"{self.BASE_URL}/api/mortys/start/"
        
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def send_morties(self, planet: int, morty_count: int) -> Dict:
        """
        Send Morties through a portal.
        
        Args:
            planet: Planet index (0 = "On a Cob", 1 = Cronenberg, 2 = Purge Planet)
            morty_count: Number of Morties to send (1-3)
            
        Returns:
            Dict with keys:
                - morties_sent: Number of Morties sent
                - survived: Whether they survived
                - morties_in_citadel: Remaining Morties in Citadel
                - morties_on_planet_jessica: Morties on Planet Jessica
                - morties_lost: Total Morties lost
                - steps_taken: Total trips taken
        """
        if planet not in [0, 1, 2]:
            raise ValueError("Planet must be 0, 1, or 2")
        
        if morty_count not in [1, 2, 3]:
            raise ValueError("Morty count must be 1, 2, or 3")
        
        url = f"{self.BASE_URL}/api/mortys/portal/"
        payload = {
            "planet": planet,
            "morty_count": morty_count
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict:
        """
        Get current episode status.
        
        Returns:
            Dict with keys:
                - morties_in_citadel: Remaining Morties in Citadel
                - morties_on_planet_jessica: Morties on Planet Jessica
                - morties_lost: Total Morties lost
                - steps_taken: Total trips taken
                - status_message: Status message
        """
        url = f"{self.BASE_URL}/api/mortys/status/"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_planet_name(self, planet_index: int) -> str:
        """
        Get the name of a planet by its index.
        
        Args:
            planet_index: Planet index (0, 1, or 2)
            
        Returns:
            Planet name
        """
        planet_names = {
            0: '"On a Cob" Planet',
            1: "Cronenberg World",
            2: "The Purge Planet"
        }
        return planet_names.get(planet_index, "Unknown Planet")


if __name__ == "__main__":
    # Example usage
    try:
        client = SphinxAPIClient()
        print("API Client initialized successfully!")
        print("Testing connection...")
        
        status = client.get_status()
        print(f"\nCurrent Status:")
        print(f"  Morties in Citadel: {status['morties_in_citadel']}")
        print(f"  Morties on Planet Jessica: {status['morties_on_planet_jessica']}")
        print(f"  Morties Lost: {status['morties_lost']}")
        print(f"  Steps Taken: {status['steps_taken']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease set up your API token:")
        print("1. Get a token from https://challenge.sphinxhq.com/")
        print("2. Create a .env file with: SPHINX_API_TOKEN=your_token_here")
    except Exception as e:
        print(f"Error: {e}")
