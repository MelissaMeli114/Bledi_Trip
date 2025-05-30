import json
import numpy as np
import random
import matplotlib.pyplot as plt
import folium
import requests 
from datetime import datetime, timedelta
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
import time

class ItineraryOptimizer:
    def __init__(self, transport_file: str, recommendation_file: str, 
                 max_time: float = 480, meal_time: float = 30, 
                 meal_travel_time: float = 30, max_walking_time: float = 60,
                 ant_count: int = 50, iterations: int = 100,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1, q0: float = 0.9,
                 start_time: str = "09:00", start_date: str = None, language: str = "en"):
        """
        Initialize optimizer with data from both JSON files
        """
        self.language = language  # 'en' or 'fr'
        
        # Performance metrics
        self.execution_time = 0
        self.iterations_completed = 0
        self.solutions_generated = 0
        self.performance_metrics = {
            'time_per_iteration': [],
            'solutions_per_iteration': [],
            'quality_per_iteration': []
        }
        
        # Load and combine data from both files
        with open(transport_file, encoding='utf-8') as f:
            transport_data = json.load(f)
        with open(recommendation_file, encoding='utf-8') as f:
            recommendation_data = json.load(f)
        
        # Create unified place data structure
        self.user_location = transport_data['user_location']
        self.modes = transport_data['modes']
        self.time_matrix = np.array(transport_data['time_matrix'])
        
        # Combine place information from both files
        self.places = self.combine_place_data(transport_data['places'], recommendation_data)
        
        # Validate dimensions
        n_locations = len(self.places) + 1  # +1 for user location
        if self.time_matrix.shape != (n_locations, n_locations, len(self.modes)):
            raise ValueError(f"Time matrix dimensions invalid. Expected {(n_locations, n_locations, len(self.modes))}, got {self.time_matrix.shape}")
        
        # Problem parameters
        self.max_time = max_time
        self.max_walking_time = max_walking_time
        self.meal_params = {
            'duration': meal_time,
            'travel_time': meal_travel_time,
            'window_start': 11.5,  # 11:30 in hours
            'window_end': 13.0     # 13:00 in hours
        }
        
        # ACO parameters
        self.ant_count = ant_count
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((len(self.places)+1, len(self.places)+1))
        
        # Heuristic matrix
        self.heuristic = self.calculate_heuristic_matrix()
        
        # Pareto front storage
        self.pareto_front = []
        
        # Schedule parameters
        self.start_time = datetime.strptime(start_time, "%H:%M")
        
        # Parse start date or use today
        if start_date:
            self.current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            self.current_date = datetime.now().date()
            
        self.full_start_time = datetime.combine(self.current_date, self.start_time.time())

        # Translation dictionary
        self.translations = {
            'en': {
                'itinerary': 'Itinerary',
                'total_time': 'Total time',
                'total_satisfaction': 'Total satisfaction',
                'places_visited': 'Places visited',
                'out_of': 'out of',
                'total_walking_time': 'Total walking time',
                'max_allowed': 'max allowed',
                'meal_break_included': 'Meal break included',
                'schedule': 'Schedule',
                'starting_at': 'Starting at',
                'travel': 'Travel',
                'from': 'from',
                'to': 'to',
                'by': 'by',
                'visit': 'Visit',
                'rating': 'Rating',
                'category': 'Category',
                'entry': 'Entry',
                'meal_break': 'Meal break',
                'min_travel': 'min travel',
                'user_location': 'User Location',
                'minutes': 'min',
                'no_solution': 'No valid solution to print',
                'pareto_title': 'Pareto Optimal Solutions',
                'time_used': 'Total Time Used (min)',
                'satisfaction_score': 'Total Satisfaction Score',
                'best_time': 'Best Time',
                'best_satisfaction': 'Best Satisfaction',
                'map_legend': 'Transportation',
                'walking': 'Walking',
                'driving': 'Driving',
                'public_transport': 'Public Transport',
                'cycling': 'Cycling',
                'taxi': 'Taxi',
                'no_solutions': 'No solutions found to visualize',
                'exported_to': 'Solution exported to',
                'map_saved_to': 'Map visualization saved to',
                'time_not_open': 'Not open at visit time',
                'day_not_open': 'Closed on visit day',
                'opening_hours': 'Opening hours',
                'opening_days': 'Opening days',
                'performance_metrics': 'Performance Metrics',
                'execution_time': 'Execution Time (s)',
                'iterations_completed': 'Iterations Completed',
                'solutions_generated': 'Solutions Generated',
                'avg_solutions_per_iter': 'Avg Solutions per Iteration',
                'avg_time_per_iter': 'Avg Time per Iteration (ms)',
                'pareto_solutions': 'Pareto Solutions Found',
                'convergence': 'Convergence Metrics',
                'time_to_first_solution': 'Time to First Solution (s)',
                'iteration_first_solution': 'Iteration of First Solution'
            },
            'fr': {
                'itinerary': 'Itinéraire',
                'total_time': 'Temps total',
                'total_satisfaction': 'Satisfaction totale',
                'places_visited': 'Lieux visités',
                'out_of': 'sur',
                'total_walking_time': 'Temps total de marche',
                'max_allowed': 'maximum autorisé',
                'meal_break_included': 'Pause repas incluse',
                'schedule': 'Horaire',
                'starting_at': 'Début à',
                'travel': 'Trajet',
                'from': 'de',
                'to': 'à',
                'by': 'par',
                'visit': 'Visite',
                'rating': 'Note',
                'category': 'Catégorie',
                'entry': 'Entrée',
                'meal_break': 'Pause repas',
                'min_travel': 'min de trajet',
                'user_location': 'Emplacement Utilisateur',
                'minutes': 'min',
                'no_solution': 'Aucune solution valide à afficher',
                'pareto_title': 'Solutions Pareto Optimales',
                'time_used': 'Temps Total Utilisé (min)',
                'satisfaction_score': 'Score de Satisfaction Total',
                'best_time': 'Meilleur Temps',
                'best_satisfaction': 'Meilleure Satisfaction',
                'map_legend': 'Transport',
                'walking': 'Marche',
                'driving': 'Voiture',
                'public_transport': 'Transport Public',
                'cycling': 'Vélo',
                'taxi': 'Taxi',
                'no_solutions': 'Aucune solution trouvée à visualiser',
                'exported_to': 'Solution exportée vers',
                'map_saved_to': 'Visualisation de la carte sauvegardée sous',
                'time_not_open': 'Fermé à l\'heure de visite',
                'day_not_open': 'Fermé le jour de visite',
                'opening_hours': 'Heures d\'ouverture',
                'opening_days': 'Jours d\'ouverture',
                'performance_metrics': 'Mesures de Performance',
                'execution_time': 'Temps d\'Exécution (s)',
                'iterations_completed': 'Itérations Complétées',
                'solutions_generated': 'Solutions Générées',
                'avg_solutions_per_iter': 'Moyenne Solutions par Itération',
                'avg_time_per_iter': 'Temps Moyen par Itération (ms)',
                'pareto_solutions': 'Solutions Pareto Trouvées',
                'convergence': 'Métriques de Convergence',
                'time_to_first_solution': 'Temps jusqu\'à Première Solution (s)',
                'iteration_first_solution': 'Itération de la Première Solution'
            }
        }

    def t(self, key: str) -> str:
        """Get translation for the given key"""
        return self.translations[self.language].get(key, key)

    def combine_place_data(self, transport_places, recommendation_data) -> List[Dict]:
        """Combine data from both JSON files into unified place structure"""
        places = []
        
        # Create mapping by name (assuming names match between files)
        rec_map = {item.get('name', ''): item for item in recommendation_data}
        
        for place in transport_places:
            # Only use values from recommendation data without defaults
            combined = {
                'name': place['name'],
                'location': place['location'],
                'rating': place['rating']
            }
            
            # Get required data from recommendation data
            if place['name'] in rec_map:
                rec = rec_map[place['name']]
                
                # Set default visit time if not provided
                combined['visit_time_min'] = rec.get('average_visit_time', 1.0) * 60
                
                # Add opening hours and days
                combined['opening_days'] = rec.get('opening_days', '')
                combined['opening_hours'] = rec.get('opening_hours', {})
                
                # Add other data from recommendations
                combined.update({
                    'category': rec.get('category', ''),
                    'characteristics': rec.get('characteristics', ''),
                    'entry_fee': rec.get('entry_fee', ''),
                    'images': rec.get('images', '')
                })
            else:
                # Default values if place not found in recommendations
                combined['visit_time_min'] = 60  # Default 1 hour
                combined['opening_days'] = ''
                combined['opening_hours'] = {}
                combined['category'] = ''
                combined['characteristics'] = ''
                combined['entry_fee'] = ''
            
            places.append(combined)
        return places

    def is_place_open_on_day(self, place: Dict, visit_date: datetime) -> bool:
        """Check if place is open on the given day of week"""
        if not place.get('opening_days'):
            return True  # Assume open every day if no data
            
        # Get day name in French (since your data appears to be in French)
        day_names_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        day_name = day_names_fr[visit_date.weekday()]
        
        return day_name in place['opening_days']

    def is_place_open_at_time(self, place: Dict, visit_time: datetime) -> bool:
        """Check if place is open at the given time"""
        if not place.get('opening_hours'):
            return True  # Assume always open if no data
            
        open_time = place['opening_hours'].get('open')
        close_time = place['opening_hours'].get('close')
        
        if not open_time or not close_time:
            return True  # Assume always open if no specific hours
            
        try:
            # Parse opening hours (handling different time formats)
            if isinstance(open_time, float):
                # Handle fractional hours (like in your example data)
                open_h = int(open_time)
                open_m = int((open_time - open_h) * 60)
                open_dt = datetime(visit_time.year, visit_time.month, visit_time.day, open_h, open_m)
            else:
                open_dt = datetime.strptime(open_time, "%H:%M:%S").time()
                open_dt = datetime.combine(visit_time.date(), open_dt)
                
            if isinstance(close_time, float):
                close_h = int(close_time)
                close_m = int((close_time - close_h) * 60)
                close_dt = datetime(visit_time.year, visit_time.month, visit_time.day, close_h, close_m)
            else:
                close_dt = datetime.strptime(close_time, "%H:%M:%S").time()
                close_dt = datetime.combine(visit_time.date(), close_dt)
                
            return open_dt <= visit_time <= close_dt
        except:
            return True  # If parsing fails, assume open

    def calculate_heuristic_matrix(self) -> np.ndarray:
        """Create heuristic matrix using ratings and travel times"""
        heuristic = np.zeros((len(self.places)+1, len(self.places)+1))
        ratings = np.array([p['rating'] for p in self.places])
        
        for i in range(heuristic.shape[0]):
            for j in range(heuristic.shape[1]):
                if i == j or j == 0:  # Skip self and user location
                    heuristic[i][j] = 0
                else:
                    place_idx = j-1
                    min_time = max(np.min(self.time_matrix[i][j]), 0.1)  # Avoid division by zero
                    heuristic[i][j] = ratings[place_idx] / min_time
        return heuristic

    def optimize(self) -> List[Dict]:
        """Run the ACO optimization"""
        start_time = time.time()
        first_solution_time = None
        first_solution_iteration = None
        
        for iteration in range(self.iterations):
            iter_start = time.time()
            ants = [self.create_ant() for _ in range(self.ant_count)]
            
            for ant in ants:
                self.build_solution(ant)
                self.evaluate_solution(ant)
                self.solutions_generated += 1
                
                # Track first solution found
                if ant['path'] and len(ant['path']) > 1 and first_solution_time is None:
                    first_solution_time = time.time() - start_time
                    first_solution_iteration = iteration
            
            self.update_pheromones(ants)
            self.update_pareto_front(ants)
            
            # Track performance metrics
            iter_time = time.time() - iter_start
            self.performance_metrics['time_per_iteration'].append(iter_time)
            self.performance_metrics['solutions_per_iteration'].append(len(ants))
            if self.pareto_front:
                avg_quality = np.mean([self.solution_quality(sol) for sol in self.pareto_front])
                self.performance_metrics['quality_per_iteration'].append(avg_quality)
            
            if (iteration+1) % 10 == 0:
                print(f"Iteration {iteration+1}: {len(self.pareto_front)} Pareto solutions")
        
        self.execution_time = time.time() - start_time
        self.iterations_completed = self.iterations
        
        # Store convergence metrics
        if first_solution_time is not None:
            self.performance_metrics['time_to_first_solution'] = first_solution_time
            self.performance_metrics['iteration_first_solution'] = first_solution_iteration
        
        return self.pareto_front

    def create_ant(self) -> Dict:
        """Initialize an ant with empty solution"""
        return {
            'path': [0],  # Start at user location
            'visited': set(),
            'time_used': 0,
            'satisfaction': 0,
            'meal_taken': False,
            'transport_modes': [],
            'walking_time': 0,
            'schedule': []
        }

    def build_solution(self, ant: Dict):
        """Construct a complete solution for an ant"""
        while True:
            current = ant['path'][-1]
            possible_moves = self.get_possible_moves(ant)
            
            if not possible_moves:
                break
                
            next_node = self.select_next_node(current, possible_moves)
            
            # Check if we should take meal break (between 11:30 and 13:00)
            current_time = self.full_start_time + timedelta(minutes=ant['time_used'])
            current_hour = current_time.hour + current_time.minute/60
            
            if (not ant['meal_taken'] and 
                self.meal_params['window_start'] <= current_hour <= self.meal_params['window_end']):
                if self.add_meal_break(ant):
                    continue
                
            self.add_move(ant, current, next_node)

    def get_possible_moves(self, ant: Dict) -> List[int]:
        """Get valid next locations considering constraints"""
        possible = []
        current = ant['path'][-1]
        
        for j in range(1, len(self.places)+1):
            if j-1 not in ant['visited'] and self.is_move_valid(ant, current, j):
                possible.append(j)
        return possible

    def is_move_valid(self, ant: Dict, i: int, j: int) -> bool:
        """Check if move satisfies all constraints"""
        # Get fastest transport mode
        mode_idx = np.argmin(self.time_matrix[i][j])
        travel_time = self.time_matrix[i][j][mode_idx]
        mode = self.modes[mode_idx]
        
        # Check walking constraint
        if mode == 'walking':
            new_walk_time = ant['walking_time'] + travel_time
            if new_walk_time > self.max_walking_time:
                return False
        
        # Check total time constraint
        place = self.places[j-1]
        visit_time = place['visit_time_min']
        total_time = ant['time_used'] + travel_time + visit_time
        if total_time > self.max_time:
            return False
            
        # Check opening hours and days
        visit_start_time = self.full_start_time + timedelta(minutes=ant['time_used'] + travel_time)
        visit_end_time = visit_start_time + timedelta(minutes=visit_time)
        
        # Check if place is open on the visit day
        if not self.is_place_open_on_day(place, visit_start_time):
            return False
            
        # Check if place is open during visit time
        if not (self.is_place_open_at_time(place, visit_start_time) and 
                self.is_place_open_at_time(place, visit_end_time)):
            return False
        
        return True

    def select_next_node(self, current: int, possible: List[int]) -> int:
        """ACO node selection with exploration/exploitation balance"""
        if not possible:
            return None
            
        # Calculate probabilities
        probs = []
        for j in possible:
            tau = self.pheromone[current][j] ** self.alpha
            eta = self.heuristic[current][j] ** self.beta
            probs.append(tau * eta)
        
        total = sum(probs)
        if total == 0:
            return random.choice(possible)
            
        probs = [p/total for p in probs]
        
        # q0 probability of greedy selection
        if random.random() < self.q0:
            return possible[np.argmax(probs)]
        else:
            return random.choices(possible, weights=probs, k=1)[0]

    def add_meal_break(self, ant: Dict) -> bool:
        """Attempt to add meal break if time permits"""
        required_time = self.meal_params['duration'] + self.meal_params['travel_time']
        if ant['time_used'] + required_time > self.max_time:
            return False
            
        ant['time_used'] += required_time
        ant['meal_taken'] = True
        
        # Add to schedule
        start_time = self.full_start_time + timedelta(minutes=ant['time_used'] - required_time)
        end_time = start_time + timedelta(minutes=required_time)
        
        ant['schedule'].append({
            'type': 'meal',
            'start': start_time.strftime("%H:%M"),
            'end': end_time.strftime("%H:%M"),
            'duration': self.meal_params['duration'],
            'travel_time': self.meal_params['travel_time']
        })
        
        return True

    def add_move(self, ant: Dict, i: int, j: int):
        """Add a move to the ant's solution"""
        # Get fastest transport mode
        mode_idx = np.argmin(self.time_matrix[i][j])
        travel_time = self.time_matrix[i][j][mode_idx]
        mode = self.modes[mode_idx]
        visit_time = self.places[j-1]['visit_time_min']
        
        # Update walking time if applicable
        if mode == 'walking':
            ant['walking_time'] += travel_time
        
        # Update ant state
        ant['path'].append(j)
        ant['visited'].add(j-1)
        ant['time_used'] += travel_time + visit_time
        ant['satisfaction'] += self.places[j-1]['rating']
        ant['transport_modes'].append(mode)
        
        # Add to schedule
        start_time = self.full_start_time + timedelta(minutes=ant['time_used'] - travel_time - visit_time)
        travel_end_time = start_time + timedelta(minutes=travel_time)
        visit_end_time = travel_end_time + timedelta(minutes=visit_time)
        
        from_location = self.user_location if i == 0 else self.places[i-1]['location']
        to_location = self.places[j-1]['location']
        
        ant['schedule'].append({
            'type': 'travel',
            'mode': mode,
            'start': start_time.strftime("%H:%M"),
            'end': travel_end_time.strftime("%H:%M"),
            'duration': travel_time,
            'from': self.get_location_name(i),
            'to': self.get_location_name(j),
            'from_location': from_location,
            'to_location': to_location
        })
        
        if j != 0:  # Not return to user location
            place = self.places[j-1]
            ant['schedule'].append({
                'type': 'visit',
                'place': place['name'],
                'start': travel_end_time.strftime("%H:%M"),
                'end': visit_end_time.strftime("%H:%M"),
                'duration': visit_time,
                'location': place['location'],
                'category': place.get('category', ''),
                'rating': place.get('rating', 0),
                'opening_days': place.get('opening_days', ''),
                'opening_hours': place.get('opening_hours', {})
            })

    def evaluate_solution(self, ant: Dict):
        """Calculate final solution metrics"""
        # Add return to user location if time permits
        current = ant['path'][-1]
        return_time = np.min(self.time_matrix[current][0])
        if ant['time_used'] + return_time <= self.max_time:
            ant['time_used'] += return_time
            ant['path'].append(0)
            # Add transport mode for return journey
            mode_idx = np.argmin(self.time_matrix[current][0])
            mode = self.modes[mode_idx]
            ant['transport_modes'].append(mode)
            
            # Add return trip to schedule
            start_time = self.full_start_time + timedelta(minutes=ant['time_used'] - return_time)
            end_time = start_time + timedelta(minutes=return_time)
            
            from_location = self.places[current-1]['location'] if current != 0 else self.user_location
            
            ant['schedule'].append({
                'type': 'travel',
                'mode': mode,
                'start': start_time.strftime("%H:%M"),
                'end': end_time.strftime("%H:%M"),
                'duration': return_time,
                'from': self.get_location_name(current),
                'to': self.t('user_location'),
                'from_location': from_location,
                'to_location': self.user_location
            })

    def update_pheromones(self, ants: List[Dict]):
        """Update pheromone trails based on Pareto solutions"""
        # Evaporation
        self.pheromone *= (1 - self.rho)
        
        # Get non-dominated solutions
        non_dominated = [a for a in ants if self.is_non_dominated(a, ants)]
        
        # Deposit pheromones
        for ant in non_dominated:
            quality = self.solution_quality(ant)
            for i, j in zip(ant['path'][:-1], ant['path'][1:]):
                self.pheromone[i][j] += quality

    def is_non_dominated(self, ant: Dict, population: List[Dict]) -> bool:
        """Check if solution is non-dominated in population"""
        for other in population:
            if (other['time_used'] < ant['time_used'] and 
                other['satisfaction'] >= ant['satisfaction']):
                return False
        return True

    def solution_quality(self, ant: Dict) -> float:
        """Calculate solution quality metric (0-1)"""
        if self.max_time == 0:  # Prevent division by zero
            time_norm = 0
        else:
            time_norm = 1 - (ant['time_used'] / self.max_time)
            
        total_possible_satisfaction = sum(p['rating'] for p in self.places)
        if total_possible_satisfaction == 0:  # Prevent division by zero
            sat_norm = 0
        else:
            sat_norm = ant['satisfaction'] / total_possible_satisfaction
            
        return 0.7 * sat_norm + 0.3 * time_norm

    def update_pareto_front(self, ants: List[Dict]):
        """Maintain global Pareto front"""
        candidates = self.pareto_front + ants
        self.pareto_front = [a for a in candidates if self.is_non_dominated(a, candidates)]
        
        # Remove duplicates
        seen = set()
        new_front = []
        for sol in self.pareto_front:
            key = (tuple(sol['path']), sol['meal_taken'])
            if key not in seen:
                seen.add(key)
                new_front.append(sol)
        self.pareto_front = new_front

    def print_performance_metrics(self):
        """Print performance metrics of the optimization"""
        print(f"\n{' ' + self.t('performance_metrics') + ' ':=^80}")
        print(f"{self.t('execution_time')}: {self.execution_time:.2f}")
        print(f"{self.t('iterations_completed')}: {self.iterations_completed}")
        print(f"{self.t('solutions_generated')}: {self.solutions_generated}")
        
        if self.iterations_completed > 0:
            avg_solutions = self.solutions_generated / self.iterations_completed
            avg_time = np.mean(self.performance_metrics['time_per_iteration']) * 1000
            print(f"{self.t('avg_solutions_per_iter')}: {avg_solutions:.1f}")
            print(f"{self.t('avg_time_per_iter')}: {avg_time:.1f}")
        
        print(f"{self.t('pareto_solutions')}: {len(self.pareto_front)}")
        
        if 'time_to_first_solution' in self.performance_metrics:
            print(f"\n{self.t('convergence')}:")
            print(f"{self.t('time_to_first_solution')}: {self.performance_metrics['time_to_first_solution']:.2f}")
            print(f"{self.t('iteration_first_solution')}: {self.performance_metrics['iteration_first_solution']}")
        
        print(f"{'':=^80}")

    def visualize_results(self):
        """Create visualization of Pareto front"""
        if not self.pareto_front:
            print(self.t('no_solutions'))
            return
            
        times = [s['time_used'] for s in self.pareto_front]
        sats = [s['satisfaction'] for s in self.pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(times, sats, c='blue', alpha=0.7)
        plt.title(self.t('pareto_title'))
        plt.xlabel(self.t('time_used'))
        plt.ylabel(self.t('satisfaction_score'))
        plt.grid(True)
        
        # Highlight best solutions
        if len(times) > 0:
            min_time_idx = np.argmin(times)
            plt.scatter(times[min_time_idx], sats[min_time_idx], 
                        c='red', s=100, label=self.t('best_time'))
            
            max_sat_idx = np.argmax(sats)
            plt.scatter(times[max_sat_idx], sats[max_sat_idx], 
                        c='green', s=100, label=self.t('best_satisfaction'))
            
            plt.legend()
        
        plt.show()

    def print_solution(self, solution: Dict):
        """Print human-readable itinerary with schedule"""
        if not solution or 'path' not in solution:
            print(self.t('no_solution'))
            return
            
        print(f"\n{' ' + self.t('itinerary') + ' ':=^80}")
        print(f"{self.t('total_time')}: {solution['time_used']:.1f} {self.t('minutes')}")
        print(f"{self.t('total_satisfaction')}: {solution['satisfaction']:.2f}")
        print(f"{self.t('places_visited')}: {len(solution['visited'])} {self.t('out_of')} {len(self.places)}")
        print(f"{self.t('total_walking_time')}: {solution.get('walking_time', 0):.1f} {self.t('minutes')} ({self.t('max_allowed')} {self.max_walking_time} {self.t('minutes')})")
        if solution['meal_taken']:
            print(self.t('meal_break_included'))
        print()
        
        print(f"{' ' + self.t('schedule') + ' ':-^80}")
        print(f"{self.t('starting_at')}: {self.start_time.strftime('%H:%M')}")
        
        for i, event in enumerate(solution['schedule'], 1):
            if event['type'] == 'travel':
                print(f"{i:2}. {event['start']} - {event['end']}: {self.t('travel')} {self.t('from')} {event['from']} {self.t('to')} {event['to']} "
                      f"({event['duration']:.1f} {self.t('minutes')} {self.t('by')} {event['mode']})")
            elif event['type'] == 'visit':
                place = next(p for p in self.places if p['name'] == event['place'])
                print(f"{i:2}. {event['start']} - {event['end']}: {self.t('visit')} {event['place']} "
                      f"({event['duration']:.1f} {self.t('minutes')}) | {self.t('rating')}: {place['rating']:.1f} | "
                      f"{self.t('category')}: {place['category']} | {self.t('entry')}: {place['entry_fee']}")
                
                # Print opening information if available
                if 'opening_days' in event and event['opening_days']:
                    print(f"   {self.t('opening_days')}: {event['opening_days']}")
                if 'opening_hours' in event and event['opening_hours']:
                    hours = event['opening_hours']
                    if hours.get('open') and hours.get('close'):
                        print(f"   {self.t('opening_hours')}: {hours['open']} - {hours['close']}")
            elif event['type'] == 'meal':
                print(f"{i:2}. {event['start']} - {event['end']}: {self.t('meal_break')} "
                      f"({event['duration']:.1f} {self.t('minutes')} + {event['travel_time']:.1f} {self.t('min_travel')})")
        
        print(f"{'':=^80}")

    def get_location_name(self, index: int) -> str:
        """Get place name from matrix index"""
        if index == 0:
            return self.t('user_location')
        return self.places[index-1]['name']
        
    def get_best_solutions(self):
        """Return the best solutions based on time and satisfaction"""
        if not self.pareto_front:
            return None, None
            
        best_time = min(self.pareto_front, key=lambda x: x['time_used'])
        best_sat = max(self.pareto_front, key=lambda x: x['satisfaction'])
        
        return best_time, best_sat
        
    def export_solution_to_json(self, solution, filename):
        """Export a solution to JSON format"""
        if not solution:
            print(self.t('no_solution'))
            return
            
        output = {
            "total_time": solution["time_used"],
            "total_satisfaction": solution["satisfaction"],
            "meal_included": solution["meal_taken"],
            "total_walking_time": solution.get("walking_time", 0),
            "itinerary": solution['schedule']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"{self.t('exported_to')} {filename}")

    def get_route_between_points(self, from_coord, to_coord, mode='walking'):
        """
        Get route between two points using OpenStreetMap's OSRM service
        
        Parameters:
        - from_coord: (lat, lon) of starting point
        - to_coord: (lat, lon) of ending point
        - mode: 'walking', 'driving', 'public_transport', 'cycling', 'taxi'
        
        Returns:
        - List of [lon, lat] coordinates for the route
        """
        # Map our transport modes to OSRM profiles
        mode_mapping = {
            'walking': 'foot',
            'driving': 'car',
            'public_transport': 'car',  # OSRM doesn't have PT routing
            'cycling': 'bike',
            'taxi': 'car'
        }
        
        # Use the appropriate OSRM profile
        profile = mode_mapping.get(mode, 'foot')
        
        # Construct the OSRM API URL
        from_lat, from_lon = from_coord
        to_lat, to_lon = to_coord
        
        url = f"http://router.project-osrm.org/route/v1/{profile}/{from_lon},{from_lat};{to_lon},{to_lat}"
        params = {
            'overview': 'full',
            'geometries': 'geojson',
            'steps': 'true'
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                print(f"Warning: OSRM API returned status code {response.status_code}")
                # Fallback to straight line
                return [[from_lon, from_lat], [to_lon, to_lat]]
                
            data = response.json()
            
            if data.get('code') != 'Ok' or 'routes' not in data or len(data['routes']) == 0:
                print(f"Warning: Could not get route from OSRM: {data.get('code', 'Unknown error')}")
                # Fallback to straight line if routing fails
                return [[from_lon, from_lat], [to_lon, to_lat]]
            
            # Extract the route coordinates
            route_geometry = data['routes'][0]['geometry']['coordinates']
            return route_geometry
            
        except Exception as e:
            print(f"Error fetching route: {e}")
            # Fallback to straight line if API call fails
            return [[from_lon, from_lat], [to_lon, to_lat]]

    def visualize_on_map(self, solution: Dict, output_file: str = "itinerary_map.html"):
        """Create an interactive map visualization of the itinerary with routes following roads"""
        if not solution or 'schedule' not in solution:
            print(self.t('no_solution'))
            return
            
        # Create base map centered on user location
        user_lat, user_lng = self.user_location
        itinerary_map = folium.Map(location=[user_lat, user_lng], zoom_start=13)
        
        # Add user location marker
        folium.Marker(
            location=[user_lat, user_lng],
            popup=f"Start/End: {self.t('user_location')}",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(itinerary_map)
        
        # Track all locations visited in order
        travel_segments = []
        for event in solution['schedule']:
            if event['type'] == 'travel':
                travel_segments.append({
                    'from_name': event['from'],
                    'to_name': event['to'],
                    'from_location': event.get('from_location', self.user_location),
                    'to_location': event.get('to_location', self.user_location),
                    'mode': event.get('mode', 'walking')
                })