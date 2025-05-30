import requests
import numpy as np
from typing import List, Dict, Tuple

class TimeMatrixCalculator:
    """Calculate travel time matrix using OpenRoute Service API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "5b3ce3597851110001cf6248624a23ad34214d319e9e4a33ec00eebb"  # Using provided API key directly
        self.base_url = "https://api.openrouteservice.org/v2/matrix/"
        self.headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml',
            'Content-Type': 'application/json',
            'Authorization': self.api_key
        }
        
    def calculate_time_matrix(self, locations: List[List[float]], modes: List[str]) -> np.ndarray:
        """
        Calculate time matrix for all locations and transport modes
        
        Args:
            locations: List of [lat, lng] coordinates
            modes: List of transport modes ('walking', 'driving', 'cycling')
            
        Returns:
            3D numpy array: [from_idx, to_idx, mode_idx] with travel times in minutes
        """
        n_locations = len(locations)
        n_modes = len(modes)
        
        # Initialize time matrix with a default value (60 minutes)
        time_matrix = np.full((n_locations, n_locations, n_modes), 60.0)
        
        # For each mode, calculate the time matrix
        for mode_idx, mode in enumerate(modes):
            # Map our transport modes to ORS profiles
            profile = self._get_ors_profile(mode)
            
            try:
                # Calculate matrix for this mode
                mode_matrix = self._calculate_matrix(locations, profile)
                
                # Store in the 3D matrix
                for i in range(n_locations):
                    for j in range(n_locations):
                        if i == j:
                            time_matrix[i, j, mode_idx] = 0
                        else:
                            # Convert seconds to minutes and round to nearest minute
                            time_matrix[i, j, mode_idx] = max(1, round(mode_matrix[i][j] / 60))
            except Exception as e:
                print(f"Error calculating matrix for mode {mode}: {str(e)}")
                # Fallback to default values if API fails
        
        return time_matrix
    
    def _get_ors_profile(self, mode: str) -> str:
        """Map our transport modes to ORS profiles"""
        mode_mapping = {
            'walking': 'foot-walking',
            'driving': 'driving-car',
            'cycling': 'cycling-regular'
        }
        return mode_mapping.get(mode, 'foot-walking')
    
    def _calculate_matrix(self, locations: List[List[float]], profile: str) -> List[List[float]]:
        """
        Calculate time matrix for given locations and profile using ORS API
        
        Returns:
            2D list of travel times in seconds
        """
        # Prepare coordinates in the format ORS expects: [lng, lat] pairs
        coordinates = [[loc[1], loc[0]] for loc in locations]
        
        # Prepare the request body
        body = {
            "locations": coordinates,
            "metrics": ["duration"],
            "sources": list(range(len(coordinates))),  # Calculate from all points
            "destinations": list(range(len(coordinates))),  # Calculate to all points
            "units": "m"
        }
        
        url = f"{self.base_url}{profile}"
        
        try:
            response = requests.post(url, json=body, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Extract durations from response
            data = response.json()
            durations = data.get("durations", [])
            
            if not durations:
                raise ValueError("Empty durations matrix returned from API")
                
            return durations
            
        except Exception as e:
            print(f"Error calling OpenRoute Service API: {str(e)}")
            raise  # Re-raise the exception to handle it in the calling function