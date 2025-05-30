from flask import render_template, request, redirect, url_for, session, jsonify, flash
import time
import json
import os
import tempfile
import geocoder
from pymongo import MongoClient
import numpy as np
from datetime import datetime, timedelta
import folium
from folium.plugins import AntPath
import random
from .time_matrix_calculator import TimeMatrixCalculator

def register_itinerary_routes(app, db):
    lieux_collection = db['lieux']

    @app.route('/get-places-info')
    def get_places_info():
        """Get information about selected places"""
        if 'user' not in session:
            return jsonify([])
            
        place_ids = request.args.get('ids', '').split(',')
        places_info = []
        
        try:
            for place_id in place_ids:
                if place_id:  # Skip empty strings
                    place = lieux_collection.find_one({"_id": int(place_id)})
                    if place:
                        places_info.append({
                            "id": place["_id"],
                            "name": place.get("name", "Unknown Place"),
                            "category": place.get("category", ""),
                            "average_visit_time": place.get("average_visit_time", 60)  # Default to 60 minutes if not specified
                        })
        except Exception as e:
            print(f"Error fetching place info: {str(e)}")
            
        return jsonify(places_info)

    @app.route('/itinerary-settings')
    def itinerary_settings():
        """Display itinerary settings form"""
        if 'user' not in session:
            flash("Please login first", "error")
            return redirect(url_for('compte'))
            
        if 'selected_places' not in session:
            flash("Please select places first", "error")
            return redirect(url_for('recommendations2'))
            
        # Get place information for display
        selected_places = []
        for place_id in session['selected_places']:
            place = lieux_collection.find_one({"_id": int(place_id)})
            if place:
                selected_places.append({
                    "id": place["_id"],
                    "name": place.get("name", "Unknown Place"),
                    "category": place.get("category", ""),
                    "average_visit_time": place.get("average_visit_time", 60)  # Default to 60 minutes
                })
            
        return render_template('itinerary_settings.html',
                            place_ids=','.join(session['selected_places']),
                            selected_places=selected_places,
                            user={"prenom": session.get('user', 'User')})

    @app.route('/generate-itinerary', methods=['POST'])
    def generate_itinerary():
        if 'user' not in session:
            flash("Please login to generate itinerary", "error")
            return redirect(url_for('compte'))
        
        selected_places = request.form.get('selected_places')
        
        if not selected_places:
            flash("No places selected", "error")
            return redirect(url_for('recommendations2'))
            
        # Convert to list if it's a string
        if isinstance(selected_places, str):
            selected_places = selected_places.split(',')
        
        if len(selected_places) < 2:
            flash("Select at least 2 places to generate itinerary", "error")
            return redirect(url_for('recommendations2'))
        
        session['selected_places'] = selected_places
        return redirect(url_for('itinerary_settings')) 

    @app.route('/optimize-itinerary', methods=['POST'])
    def optimize_itinerary():
        if 'user' not in session or 'selected_places' not in session:
            flash("Session expired", "error")
            return redirect(url_for('recommendations2'))
        
        try:
            # Get user settings
            start_time = request.form.get('start_time', '09:00')
            end_time = request.form.get('end_time', '17:00')
            
            # Calculate max_time from start and end times
            start_dt = datetime.strptime(start_time, '%H:%M')
            end_dt = datetime.strptime(end_time, '%H:%M')
            
            # Handle end time before start time (next day)
            duration = end_dt - start_dt
            if duration.total_seconds() < 0:
                end_dt = end_dt + timedelta(days=1)
                duration = end_dt - start_dt
                
            max_time = int(duration.total_seconds() / 60)  # Convert to minutes
            
            max_walking = int(request.form.get('max_walking', 60))  # 1 hour default
            include_meal = request.form.get('include_meal') == '1'
            
            # Get selected transport modes
            transport_modes = request.form.getlist('transport_modes')
            if not transport_modes:
                transport_modes = ['walking']  # Default to walking if none selected
            
            # Get user location
            g = geocoder.ip('me')
            user_location = g.latlng if g.latlng else [36.7525, 3.0420]  # Algiers default
            
            # Get selected places data with average_visit_time
            places = []
            place_ids = session['selected_places']
            for place_id in place_ids:
                place = lieux_collection.find_one({"_id": int(place_id)})
                if place and 'coordinates' in place:
                    places.append({
                        '_id': place['_id'],
                        'name': place.get('name', 'Unknown Place'),
                        'coordinates': place['coordinates'],
                        'category': place.get('category', ''),
                        'average_visit_time': place.get('average_visit_time', 60),  # Default to 60 minutes
                        'rating': place.get('rating', 3)  # Default rating if not specified
                    })
            
            if len(places) < 2:
                flash("Not enough valid places", "error")
                return redirect(url_for('recommendations2'))
            
            # Use the TimeMatrixCalculator to get the time matrix
            # Using provided API key directly instead of environment variable
            calculator = TimeMatrixCalculator("5b3ce3597851110001cf624872d6a11db9854f3e92ab9d9c8332fe41")
            
            # Prepare locations including user location
            locations = [user_location] + [[place['coordinates']['lat'], place['coordinates']['lng']] for place in places]
            
            # Use optimized itinerary creation with the proper time matrix
            best_itinerary = create_optimized_itinerary(
                user_location=user_location,
                places=places,
                max_time=max_time,
                max_walking=max_walking,
                include_meal=include_meal,
                start_time=start_time,
                transport_modes=transport_modes,
                calculator=calculator
            )
            
            # Generate map visualization
            itinerary_map = create_itinerary_map(user_location, best_itinerary, places)
            map_html = itinerary_map._repr_html_()
            
            # Prepare timeline events
            itinerary_events = prepare_timeline_events(best_itinerary)
            
            return render_template('itinerary.html',
                user={"prenom": session.get('user', 'User')},
                itinerary=best_itinerary,
                map_html=map_html,
                itinerary_events=itinerary_events,
                best_time=best_itinerary['time_used'],
                best_satisfaction=best_itinerary['satisfaction'],
                places_visited=len(best_itinerary['visited']),
                walking_time=best_itinerary.get('walking_time', 0))
                
        except Exception as e:
            print(f"Error optimizing itinerary: {str(e)}")
            flash(f"Error generating itinerary: {str(e)}", "error")
            return redirect(url_for('itinerary_settings'))

    def create_optimized_itinerary(user_location, places, max_time, max_walking, include_meal, start_time, transport_modes, calculator):
        """
        Create an optimized itinerary using the TimeMatrixCalculator
        """
        # Initialize itinerary data
        itinerary = {
            'schedule': [],
            'visited': [],
            'time_used': 0,
            'satisfaction': 0,
            'walking_time': 0,
            'start_time': start_time
        }
        
        # Prepare locations including user location
        locations = [user_location] + [[place['coordinates']['lat'], place['coordinates']['lng']] for place in places]
        
        # Calculate time matrix using the calculator
        time_matrix = calculator.calculate_time_matrix(locations, transport_modes)
        
        # Sort places by rating for better initial solution
        sorted_places = sorted(enumerate(places), key=lambda x: x[1].get('rating', 0), reverse=True)
        
        current_location_idx = 0  # Start at user location
        current_time = datetime.strptime(start_time, '%H:%M')
        remaining_time = max_time
        remaining_walk_time = max_walking
        meal_taken = False
        
        # Add places one by one using greedy approach
        for idx, place in sorted_places:
            place_idx = idx + 1  # +1 because user location is at index 0
            
            # Skip if already visited
            if place_idx - 1 in itinerary['visited']:
                continue
                
            # Calculate travel times from current location using each transport mode
            travel_times = []
            for mode_idx, mode in enumerate(transport_modes):
                travel_time = time_matrix[current_location_idx, place_idx, mode_idx]
                
                # Skip walking if exceeds remaining walk time
                if mode == 'walking' and travel_time > remaining_walk_time:
                    continue
                    
                travel_times.append((travel_time, mode_idx))
            
            if not travel_times:
                continue  # No feasible transport mode
                
            # Select fastest mode
            travel_time, mode_idx = min(travel_times, key=lambda x: x[0])
            transport_mode = transport_modes[mode_idx]
            
            # Get place visit time from database (default to 60 minutes if not specified)
            visit_time = place.get('average_visit_time', 60)
            
            # Check if meal should be inserted before this place
            if include_meal and not meal_taken:
                meal_hour = current_time.hour + current_time.minute/60
                if 11.5 <= meal_hour <= 14.0:  # Meal window between 11:30 and 14:00
                    meal_duration = 60  # 1 hour for meal
                    if remaining_time >= meal_duration:
                        # Add meal break
                        meal_start = current_time
                        meal_end = meal_start + timedelta(minutes=meal_duration)
                        
                        itinerary['schedule'].append({
                            'type': 'meal',
                            'start': meal_start.strftime('%H:%M'),
                            'end': meal_end.strftime('%H:%M'),
                            'duration': meal_duration
                        })
                        
                        current_time = meal_end
                        remaining_time -= meal_duration
                        itinerary['time_used'] += meal_duration
                        meal_taken = True
            
            # Check if we have enough time for travel and visit
            total_activity_time = travel_time + visit_time
            if total_activity_time > remaining_time:
                continue  # Not enough time, skip this place
                
            # Update walking time if applicable
            if transport_mode == 'walking':
                remaining_walk_time -= travel_time
                itinerary['walking_time'] += travel_time
                
            # Add travel to schedule
            travel_start = current_time
            travel_end = travel_start + timedelta(minutes=travel_time)
            
            from_location = user_location if current_location_idx == 0 else [places[current_location_idx-1]['coordinates']['lat'], places[current_location_idx-1]['coordinates']['lng']]
            to_location = [place['coordinates']['lat'], place['coordinates']['lng']]
            
            itinerary['schedule'].append({
                'type': 'travel',
                'mode': transport_mode,
                'from': 'Starting Point' if current_location_idx == 0 else places[current_location_idx-1]['name'],
                'to': place['name'],
                'from_location': from_location,
                'to_location': to_location,
                'start': travel_start.strftime('%H:%M'),
                'end': travel_end.strftime('%H:%M'),
                'duration': travel_time
            })
            
            # Add visit to schedule
            visit_start = travel_end
            visit_end = visit_start + timedelta(minutes=visit_time)
            
            itinerary['schedule'].append({
                'type': 'visit',
                'place': place['name'],
                'location': [place['coordinates']['lat'], place['coordinates']['lng']],
                'category': place.get('category', ''),
                'rating': place.get('rating', 3),
                'start': visit_start.strftime('%H:%M'),
                'end': visit_end.strftime('%H:%M'),
                'duration': visit_time
            })
            
            # Update state
            current_location_idx = place_idx
            current_time = visit_end
            remaining_time -= total_activity_time
            itinerary['time_used'] += total_activity_time
            itinerary['satisfaction'] += place.get('rating', 3)
            itinerary['visited'].append(idx)
            
        # Add return to starting point if time permits and we visited at least one place
        if len(itinerary['visited']) > 0:
            return_times = []
            for mode_idx, mode in enumerate(transport_modes):
                return_time = time_matrix[current_location_idx, 0, mode_idx]
                
                # Skip walking if exceeds remaining walk time
                if mode == 'walking' and return_time > remaining_walk_time:
                    continue
                    
                return_times.append((return_time, mode_idx))
            
            if return_times:
                return_time, mode_idx = min(return_times, key=lambda x: x[0])
                return_mode = transport_modes[mode_idx]
                
                if return_time <= remaining_time:
                    # Add return travel to schedule
                    return_start = current_time
                    return_end = return_start + timedelta(minutes=return_time)
                    
                    from_location = [places[current_location_idx-1]['coordinates']['lat'], places[current_location_idx-1]['coordinates']['lng']]
                    
                    itinerary['schedule'].append({
                        'type': 'travel',
                        'mode': return_mode,
                        'from': places[current_location_idx-1]['name'],
                        'to': 'Starting Point',
                        'from_location': from_location,
                        'to_location': user_location,
                        'start': return_start.strftime('%H:%M'),
                        'end': return_end.strftime('%H:%M'),
                        'duration': return_time
                    })
                    
                    if return_mode == 'walking':
                        itinerary['walking_time'] += return_time
                        
                    itinerary['time_used'] += return_time
        
        return itinerary

    def create_itinerary_map(user_location, itinerary, places):
        """Create an interactive map for the itinerary"""
        # Create map centered on first location
        m = folium.Map(location=user_location, zoom_start=13)
        
        # Add starting point marker
        folium.Marker(
            location=user_location,
            popup='Starting Point',
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
        
        # Define colors for different transport modes
        mode_colors = {
            'walking': 'green',
            'driving': 'blue',
            'cycling': 'purple'
        }
        
        # Add markers for places and routes
        for event in itinerary['schedule']:
            if event['type'] == 'visit':
                # Add place marker
                folium.Marker(
                    location=event['location'],
                    popup=f"{event['place']} - {event['category']} (Rating: {event['rating']})<br>Duration: {event['duration']} min",
                    icon=folium.Icon(color='blue')
                ).add_to(m)
                
            elif event['type'] == 'travel':
                # Add route line with appropriate color based on mode
                color = mode_colors.get(event['mode'], 'gray')
                
                # Get coordinates
                from_loc = event['from_location']
                to_loc = event['to_location']
                
                if from_loc and to_loc:
                    # Create a simple line between points
                    folium.PolyLine(
                        locations=[from_loc, to_loc],
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f"{event['mode']} ({event['duration']} min)"
                    ).add_to(m)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; background-color:white; 
                    padding:10px; border:2px solid grey; border-radius:5px">
          <p><b>Transport Modes</b></p>
          <p><i style="background:green;width:10px;height:10px;display:inline-block"></i> Walking</p>
          <p><i style="background:blue;width:10px;height:10px;display:inline-block"></i> Driving</p>
          <p><i style="background:purple;width:10px;height:10px;display:inline-block"></i> Cycling</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m

    def prepare_timeline_events(itinerary):
        """Format itinerary events for the timeline display"""
        events = []
        
        for event in itinerary['schedule']:
            # Copy the event and format for display
            formatted_event = event.copy()
            
            # Fix from/to locations for first/last travel segments
            if event['type'] == 'travel':
                if 'Previous Location' in event['from']:
                    # Find the actual previous location name
                    prev_events = [e for e in events if e['type'] == 'visit']
                    if prev_events:
                        formatted_event['from'] = prev_events[-1]['place']
                    else:
                        formatted_event['from'] = 'Starting Point'
                if 'Last Location' in event['from']:
                    # Find the actual last visited place
                    visit_events = [e for e in events if e['type'] == 'visit']
                    if visit_events:
                        formatted_event['from'] = visit_events[-1]['place']
                    else:
                        formatted_event['from'] = 'Starting Point'
                    
            events.append(formatted_event)
            
        return events

    # Register routes
    app.add_url_rule('/get-places-info', 'get_places_info', get_places_info)
    app.add_url_rule('/itinerary-settings', 'itinerary_settings', itinerary_settings)
    app.add_url_rule('/generate-itinerary', 'generate_itinerary', generate_itinerary, methods=['POST'])
    app.add_url_rule('/optimize-itinerary', 'optimize_itinerary', optimize_itinerary, methods=['POST'])
    
    return app