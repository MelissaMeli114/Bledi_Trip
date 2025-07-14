from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import bcrypt
from datetime import datetime, timedelta
import re
from flask import send_file
from io import BytesIO
import requests
from geopy.geocoders import Nominatim
import geocoder
from datetime import date
import os
import folium

# Class for content-based recommendation
class ContentBasedRecommender:
    def __init__(self):
        self.client = MongoClient('mongodb+srv://Melissa:Melissa@cluster0.dmascbk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
        self.db = self.client["Algeria"]
        self.users = self.db["user_preferences"]
        self.lieux = self.db["lieux"]

        self.categories = {
            'Naturel': 0, 'Historique': 1, 'Culturels': 2,
            'shopping': 3, 'Loisir et Divertissement': 4,
            'Religieux': 5, 'Architectural, artificiel et patrimonial bâti': 6,
            'Bien-être / Thérapeutique': 7, 'artistiques': 8
        }

        self.subcategories = {
            'Tour': 0, 'Eglise': 1, 'Site Archeologique': 2, 'Monument': 3,
            'parc national': 4, 'foret': 5, 'Montagne': 6, 'Plage': 7,
            'Centre commercial': 8, 'Spa': 9, 'Zoo': 10, 'Parc d\'attraction': 11,
            'Pont': 12, 'musée': 13, 'cinéma': 14, 'Galerie d\'art': 15,
            'theatre': 16, 'Parc aquatique': 17, 'Parc urbain': 18, 'Village': 19,
            'Cascades': 20, 'grotte': 21, 'Lac': 22, 'Palais': 23, 'Port': 24,
            'Île': 25, 'parc naturel': 26, 'péninsule': 27, 'Barrage': 28,
            'phare': 29, 'ville': 30, 'Oasis': 31, 'Activités': 32, 'gorges': 33,
            'place urbaine': 34, 'rivière': 35, 'Dunes': 36, 'Tunnel': 37,
            'château': 38, 'Jardin': 39, 'Piscine': 40, 'Stade': 41, 'Marais': 42,
            'Fort': 43
        }

    def get_user_preferences(self, user_id):
        user_doc = self.users.find_one({'user_id': user_id})
        if not user_doc:
            return [], []

        preferences = user_doc.get('preferences', {})

        selected_categories = [
            cat for cat in self.categories if preferences.get(cat.lower(), 0) == 1
        ]
        selected_subcategories = [
            sub for sub in self.subcategories if preferences.get(sub.lower().replace(' ', '_'), 0) == 1
        ]

        return selected_categories, selected_subcategories

    def get_current_season(self):
        today = date.today()
        y = today.year

        spring = (date(y, 3, 21), date(y, 6, 20))
        summer = (date(y, 6, 21), date(y, 9, 22))
        autumn = (date(y, 9, 23), date(y, 12, 20))
        if spring[0] <= today <= spring[1]:
            return "Printemps"
        elif summer[0] <= today <= summer[1]:
            return "Eté"
        elif autumn[0] <= today <= autumn[1]:
            return "Automne"
        else:
            return "Hiver"

    def get_current_wilaya(self):
        try:
            g = geocoder.ip('me')
            latitude, longitude = g.latlng

            geolocator = Nominatim(user_agent="bleditrip_app")
            location = geolocator.reverse((latitude, longitude), language='fr')

            if location and 'address' in location.raw:
                address = location.raw['address']
                wilaya = address.get('state') or address.get('county') or address.get('region')
                return wilaya.strip() if wilaya else None
        except Exception as e:
            print("Erreur de géolocalisation :", e)
        return None

    def recommend_places_with_params(self, user_id, saison=None, wilaya=None, top_k=30):
            selected_categories, selected_subcategories = self.get_user_preferences(user_id)
            
            if not saison:
                saison = self.get_current_season()
            if not wilaya:
                wilaya = self.get_current_wilaya()

            if not selected_categories and not selected_subcategories:
                return []

            query = {
                '$and': [
                    {
                        '$or': [
                            {'category': {'$in': selected_categories}},
                            {'subcategory': {'$in': selected_subcategories}}
                        ]
                    },
                    {'best_season': {'$regex': saison, '$options': 'i'}}
                ]
            }

            if wilaya:
                query['$and'].append({'wilaya': wilaya})

            return list(self.lieux.find(query).limit(top_k))

# Import the Deep Learning recommender
from recommendations import AlgeriaTourismRecommender
from recommedations2 import AlgeriaTourismRecommender1

# Initializer the recommendation systems
recommender = ContentBasedRecommender()
# Initialize the deep learning recommender
dl_recommender = AlgeriaTourismRecommender()
# Initialize the advanced deep learning recommender
dl_recommender2 = AlgeriaTourismRecommender1()

app = Flask(__name__)
app.secret_key = '3ed235a79bbc94d041fdef6f13146e1d'

# Connexion à MongoDB
client = MongoClient('mongodb+srv://Melissa:Melissa@cluster0.dmascbk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['Algeria']
users_collection = db['users']
lieux_collection = db['lieux']
interactions_collection = db['interactions']
user_preferences_collection = db['user_preferences']

@app.route('/')
def home():
    return render_template('projet.html')

# Configuration des exigences de mot de passe
MIN_PASSWORD_LENGTH = 8
PASSWORD_REQUIREMENTS = {
    'min_length': f"Au moins {MIN_PASSWORD_LENGTH} caractères",
}
    
@app.route('/compte', methods=['GET', 'POST'])
def compte():
    if request.method == 'GET':
        return render_template('compte.html', password_reqs=PASSWORD_REQUIREMENTS)  
    
    # Détection explicite du formulaire utilisé
    is_login_form = 'login-submit' in request.form
    is_register_form = 'register-submit' in request.form

    # Traitement du formulaire de connexion
    if is_login_form:
        pseudo = request.form['pseudo'].strip()
        password = request.form['password']
        
        if not pseudo or not password:
            flash("Tous les champs sont obligatoires", "error")
            return redirect(url_for('compte'))
        
        user = users_collection.find_one({"_id": pseudo})
        
        if not user:
            flash("Nom d'utilisateur incorrect", "error")
            return redirect(url_for('compte'))

        # Vérifier si l'utilisateur est bloqué
        if user.get('failed_attempts', 0) >= 3:
            last_failed_attempt = user.get('last_failed_attempt')
            if last_failed_attempt and (datetime.now() - last_failed_attempt).total_seconds() < 30:
                flash("Trop de tentatives échouées. Veuillez patienter 30 secondes.", "error")
                return redirect(url_for('compte'))
        
        try:
            if bcrypt.checkpw(password.encode('utf-8'), user['mot_de_passe'].encode('utf-8')):
                session['user'] = pseudo
                # Vérifier s'il a des interactions OU des préférences enregistrées
                has_interactions = interactions_collection.count_documents({"user_id": pseudo}) > 0
                has_preferences = user_preferences_collection.count_documents({"user_id": pseudo}) > 0

                if not has_interactions and not has_preferences:
                    return redirect(url_for('preferences'))
                return redirect(url_for('accueil'))
            else:
                # Incrémenter le compteur de tentatives échouées
                users_collection.update_one(
                    {"_id": pseudo},
                    {"$inc": {"failed_attempts": 1}, "$set": {"last_failed_attempt": datetime.now()}}
                )
                flash("Mot de passe incorrect", "error")
                return redirect(url_for('compte'))
        except Exception as e:
            flash("Erreur lors de la connexion", "error")
            return redirect(url_for('compte'))
        

    # Traitement du formulaire d'inscription
    elif is_register_form:
        pseudo = request.form.get('pseudo', '').strip()
        password = request.form.get('passwordCreate', '')
        nom = request.form.get('names', '').strip()
        prenom = request.form.get('surnames', '').strip()
        sexe = request.form.get('sexe', '').strip()
        birthdate_str = request.form.get('birthdate', '').strip()

        errors = []

        # Validation des champs
        if not pseudo:
            errors.append("Le pseudo est obligatoire")
        elif len(pseudo) < 4:
            errors.append("Le pseudo doit contenir au moins 4 caractères")
        elif users_collection.find_one({'_id': pseudo}):
            errors.append("Ce pseudo est déjà utilisé")

        if not nom:
            errors.append("Le nom est obligatoire")
        if not prenom:
            errors.append("Le prénom est obligatoire")

        try:
            birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d')
            age = (datetime.now() - birthdate).days // 365
            if age < 13:
                errors.append("Vous devez avoir au moins 13 ans pour vous inscrire")
        except ValueError:
            errors.append("Date de naissance invalide (format attendu : AAAA-MM-JJ)")

        password_errors = check_password_strength(password)
        if password_errors:
            errors.extend(password_errors)

        if errors:
            for error in errors:
                flash(error, "error")
            return redirect(url_for('compte', _anchor='register'))

        # Insertion dans la base de données
        try:
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            user_data = {
                '_id': pseudo,
                'nom': nom,
                'prenom': prenom,
                'mot_de_passe': hashed_pw.decode('utf-8'),
                'age': age,
                'sexe': sexe,
                'visited_places': [],
                'created_at': datetime.now()
            }
            
            result = users_collection.insert_one(user_data)
            
            if result.inserted_id:
                flash("Inscription réussie! Vous pouvez maintenant vous connecter.", "success")
                return redirect(url_for('compte', _anchor='login'))
            else:
                raise Exception("Aucun ID retourné après insertion")
                
        except Exception as e:
            print("Erreur d'insertion:", str(e))
            flash(f"Erreur technique lors de l'inscription: {str(e)}", "error")
            return redirect(url_for('compte', _anchor='register'))

    return redirect(url_for('compte'))

def check_password_strength(password):
    errors = []
    password = password.strip()  # Supprime les espaces avant/après
    
    # Vérification longueur
    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères")

    if not re.search(r"[0-9]", password):
        errors.append("Le mot de passe doit contenir au moins un chiffre")
    return errors

@app.route('/profil')
def profil():
    if 'user' not in session:
        flash("Vous devez être connecté pour accéder à votre profil", "error")
        return redirect(url_for('compte'))

    user = users_collection.find_one({"_id": session['user']})
    
    # Géolocalisation
    try:
        g = geocoder.ip('me')
        latitude, longitude = g.latlng

        geolocator = Nominatim(user_agent="bleditrip_app")
        location = geolocator.reverse((latitude, longitude), language='fr')

        wilaya = None
        commune = None
        if location and 'address' in location.raw:
            address = location.raw['address']
            wilaya = address.get('state') or address.get('county') or address.get('region')
            commune = address.get('town') or address.get('village') or address.get('city')

    except Exception as e:
        print(f"Erreur géolocalisation profil : {e}")
        wilaya = None
        commune = None

    return render_template('profil.html', user=user, wilaya=wilaya, commune=commune)
@app.route('/accueil')
def accueil():
    if 'user' not in session:
        flash("Veuillez vous connecter pour accéder à l'accueil", "error")
        return redirect(url_for('compte'))

    user = users_collection.find_one({"_id": session['user']})
    
    # Récupérer toutes les wilayas disponibles
    wilayas = lieux_collection.distinct("wilaya")
    wilayas.sort()  # Trier par ordre alphabétique
    
    # Récupérer la wilaya sélectionnée (soit par paramètre, soit par géolocalisation)
    selected_wilaya = request.args.get('wilaya')
    
    try:
        g = geocoder.ip('me')
        latitude, longitude = g.latlng
        
        # Reverse geocoding pour obtenir la wilaya
        geolocator = Nominatim(user_agent="bleditrip_app")
        location = geolocator.reverse((latitude, longitude), language='fr')
        
        detected_wilaya = None
        commune = None
        if location and 'address' in location.raw:
            address = location.raw['address']
            detected_wilaya = address.get('state') or address.get('county') or address.get('region')
            commune = address.get('town') or address.get('village') or address.get('city')
        
        # Déterminer la wilaya à utiliser (priorité au filtre sélectionné)
        current_wilaya = selected_wilaya or detected_wilaya
        
        # Récupérer les lieux selon la wilaya choisie
        lieux_data = []
        if current_wilaya:
            query = {"wilaya": {"$regex": f"^{current_wilaya}$", "$options": "i"}}
            lieux = lieux_collection.find(query)
            
            for lieu in lieux:
                # Initialisation avec l'image par défaut
                image_drive_url = url_for('static', filename='images/placeholder.jpg')
                # Gestion sécurisée des images comme dans recommended_places
                if 'images' in lieu and lieu['images']:
                    try:
                        # Gère à la fois les chaînes et les listes d'images
                        image_url = lieu['images'][0] if isinstance(lieu['images'], list) else lieu['images']
                        
                        if isinstance(image_url, str) and 'id=' in image_url:
                            image_id = image_url.split('id=')[-1].split('&')[0]  # Plus sécurisé
                            image_drive_url = url_for('drive_image', file_id=image_id)
                    except Exception as e:
                        print(f"Erreur image pour lieu {lieu.get('_id', 'inconnu')}: {str(e)}")
                        # L'image par défaut est déjà définie

                # Le reste du code reste inchangé
                lieux_data.append({
                    "lieu_id": str(lieu['_id']),
                    "nom": lieu.get('name', 'Lieu inconnu'),
                    "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
                    "commune": lieu.get('commune', 'Commune inconnue'),
                    "category": lieu.get('category', 'Non catégorisé'),
                    "subcategory": lieu.get('subcategory', ''),
                    "entry_fee": lieu.get('entry_fee', ""),
                    "address": lieu.get('address', ""),
                    "image_drive_url": image_drive_url,
                    "position": (latitude, longitude),
                    "wilaya_detectee": detected_wilaya
                })
        
        return render_template('accueil.html', 
                             user=user, 
                             lieux_data=lieux_data,
                             position=(latitude, longitude),
                             wilaya=current_wilaya,
                             wilayas=wilayas,
                             current_wilaya=current_wilaya)
    
    except Exception as e:
        print(f"Erreur de géolocalisation: {str(e)}")
        # Fallback sans géolocalisation
        current_wilaya = selected_wilaya
        lieux_data = []
        
        query = {}
        if current_wilaya:
            query = {"wilaya": {"$regex": f"^{current_wilaya}$", "$options": "i"}}
        
        lieux = lieux_collection.find(query).limit(20)
        
        for lieu in lieux:
            if 'images' in lieu:
                image_url = lieu['images']
                image_id = image_url.split('id=')[-1]
                image_drive_url = url_for('drive_image', file_id=image_id)
            else:
                image_drive_url = url_for('static', filename='images/placeholder.jpg')

            lieux_data.append({
                "lieu_id": str(lieu['_id']),
                "nom": lieu.get('name', 'Lieu inconnu'),
                "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
                "commune": lieu.get('commune', 'Commune inconnue'),
                "category": lieu.get('category', 'Non catégorisé'),
                "subcategory": lieu.get('subcategory', ''),
                "entry_fee": lieu.get('entry_fee', ""),
                "address": lieu.get('address', ""),
                "image_drive_url": image_drive_url,
                "wilaya_detectee": None
            })
        
        return render_template('accueil.html', 
                             user=user, 
                             lieux_data=lieux_data,
                             position=None,
                             wilaya=current_wilaya,
                             wilayas=wilayas,
                             current_wilaya=current_wilaya)
@app.route('/details_lieu')
def details_lieu():
    if 'user' not in session:
        flash("Veuillez vous connecter pour voir les détails", "error")
        return redirect(url_for('compte'))

    lieu_id = request.args.get('lieu_id')
    if not lieu_id:
        flash("ID du lieu manquant", "error")
        return redirect(url_for('accueil'))

    try:
        lieu_id = int(lieu_id)
    except ValueError:
        flash("ID invalide", "error")
        return redirect(url_for('accueil'))

    # Récupération du lieu
    lieu = lieux_collection.find_one({"_id": lieu_id})
    if not lieu:
        flash("Lieu non trouvé", "error")
        return redirect(url_for('accueil'))

    # Image Drive
    if 'images' in lieu:
        image_url = lieu['images']
        image_id = image_url.split('id=')[-1]
        image_drive_url = url_for('drive_image', file_id=image_id)
    else:
        image_drive_url = url_for('static', filename='images/placeholder.jpg')

    # Récupération des commentaires avec pagination
    commentaires = []
    interactions = interactions_collection.find({
        "place_id": lieu_id,
        "review": {"$exists": True, "$ne": ""}  # Seulement les interactions avec commentaire
    }).sort("timestamp", -1)  # Tri par date décroissante
    
    for interaction in interactions:
        user = users_collection.find_one({"_id": interaction['user_id']})
        
        commentaires.append({
            "user_prenom": user.get('prenom', interaction['user_id']) if user else interaction['user_id'],
            "rating": interaction.get('rating', 0),
            "review": interaction['review'],
            "timestamp": interaction.get('timestamp', '')
        })

    # Récupération de l'utilisateur connecté
    current_user_doc = users_collection.find_one({"_id": session['user']})
    current_user = {
        "prenom": current_user_doc.get('prenom', 'Utilisateur')
    } if current_user_doc else {"prenom": "Utilisateur"}

    opening_hours = lieu.get('opening_hours', {})
    opening_days = lieu.get('opening_days')

    # Formater les heures si elles existent
    formatted_hours = None
    if opening_hours and opening_hours.get('open') and opening_hours.get('close'):
        try:
            open_time = opening_hours['open'].replace(':00:00', 'h') if ':00:00' in opening_hours['open'] else opening_hours['open']
            close_time = opening_hours['close'].replace(':00:00', 'h') if ':00:00' in opening_hours['close'] else opening_hours['close']
            formatted_hours = f"{open_time} à {close_time}"
        except:
            formatted_hours = None

    return render_template('details_lieu.html', 
        lieu={
            "_id": lieu['_id'],
            "name": lieu.get('name', 'Lieu inconnu'),
            "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
            "commune": lieu.get('commune', 'Commune inconnue'),
            "category": lieu.get('category', 'Non catégorisé'),
            "subcategory": lieu.get('subcategory', ''),
            "entry_fee": lieu.get('entry_fee', None),
            "address": lieu.get('address', ""),
            "history": lieu.get('history', 'Aucune information historique disponible'),
            "characteristics": lieu.get('characteristics', ''),
            "opening_days": opening_days,
            "formatted_hours": formatted_hours,
        },
        image_url=image_drive_url,
        commentaires=commentaires,
        user=current_user 
    )

@app.route('/drive-image/<file_id>')
def drive_image(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Lève une erreur HTTP si le lien est invalide
        img_data = BytesIO(response.content)
        return send_file(img_data, mimetype='image/jpeg')
    except Exception as e:
        # Retourner une image locale de secours
        return send_file('static/img/placeholder.jpg', mimetype='image/jpeg')

@app.route('/update-profile', methods=['POST'])
def update_profile():
    if 'user' not in session:
        return redirect(url_for('compte'))

    pseudo = session['user']
    user = users_collection.find_one({"_id": pseudo})

    if not user:
        flash("Utilisateur introuvable", "error")
        return redirect(url_for('profil'))

    # Récupérer les champs modifiés
    nom = request.form.get('nom', '').strip()
    prenom = request.form.get('prenom', '').strip()
    sexe = request.form.get('sexe', '').strip()

    users_collection.update_one(
        {"_id": pseudo},
        {"$set": {
            "nom": nom,
            "prenom": prenom,
            "sexe": sexe
        }}
    )

    flash("Profil mis à jour avec succès", "success")
    return redirect(url_for('profil'))

from datetime import datetime
from flask import session, request, jsonify

@app.route("/noter_lieu", methods=["POST"])
def noter_lieu():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Non connecté"}), 401

    data = request.get_json()
    lieu_id = data.get("lieu_id")
    note = data.get("note")

    try:
        note = int(note)
        lieu_id = int(lieu_id)  # Convertir en entier si nécessaire
        if note < 1 or note > 5:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Données invalides"}), 400

    user_id = session['user']
    
    try:
        # Vérifier que le lieu existe
        lieu = lieux_collection.find_one({"_id": lieu_id})
        if not lieu:
            return jsonify({"success": False, "error": "Lieu non trouvé"}), 404

        # Insertion ou mise à jour de la note
        interactions_collection.update_one(
            {"user_id": user_id, "place_id": lieu_id},
            {
                "$set": {
                    "rating": note,
                    "timestamp": datetime.utcnow()
                },
                "$setOnInsert": {
                    "review": ""
                }
            },
            upsert=True
        )

        # Ajouter le lieu aux visited_places s'il n'y est pas déjà
        users_collection.update_one(
            {"_id": user_id},
            {"$addToSet": {"visited_places": lieu_id}}
        )

        return jsonify({"success": True}), 200

    except Exception as e:
        print(f"Erreur lors de la notation: {str(e)}")
        return jsonify({"success": False, "error": "Erreur serveur"}), 500
    
@app.route('/ajouter_commentaire', methods=['POST'])
def ajouter_commentaire():
    if 'user' not in session:
        return jsonify({"success": False, "error": "Non connecté"}), 401

    data = request.get_json()
    lieu_id = data.get("lieu_id")
    commentaire = data.get("commentaire", "").strip()

    try:
        lieu_id = int(lieu_id)  # Convertir en entier si nécessaire
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "ID de lieu invalide"}), 400

    if not commentaire:
        return jsonify({"success": False, "error": "Le commentaire ne peut pas être vide"}), 400

    user_id = session['user']
    
    try:
        # Vérifier que le lieu existe
        lieu = lieux_collection.find_one({"_id": lieu_id})
        if not lieu:
            return jsonify({"success": False, "error": "Lieu non trouvé"}), 404

        # Vérifier si l'utilisateur a déjà une interaction avec ce lieu
        existing_interaction = interactions_collection.find_one({
            "user_id": user_id,
            "place_id": lieu_id
        })

        if existing_interaction:
            # Mise à jour du commentaire existant
            interactions_collection.update_one(
                {"user_id": user_id, "place_id": lieu_id},
                {
                    "$set": {
                        "review": commentaire,
                        "timestamp": datetime.utcnow()
                    }
                }
            )
        else:
            # Création d'une nouvelle interaction avec commentaire mais sans note
            new_interaction = {
                "user_id": user_id,
                "place_id": lieu_id,
                "review": commentaire,
                "timestamp": datetime.utcnow()
            }
            interactions_collection.insert_one(new_interaction)

        # Ajouter le lieu aux visited_places s'il n'y est pas déjà
        users_collection.update_one(
            {"_id": user_id},
            {"$addToSet": {"visited_places": lieu_id}}
        )

        # Récupérer les infos de l'utilisateur pour l'affichage
        user = users_collection.find_one({"_id": user_id})
        user_prenom = user.get('prenom', user_id) if user else user_id

        return jsonify({
            "success": True,
            "commentaire": {
                "user_prenom": user_prenom,
                "rating": existing_interaction.get('rating', 0) if existing_interaction else 0,
                "review": commentaire,
                "timestamp": datetime.utcnow().isoformat()
            }
        }), 200

    except Exception as e:
        print(f"Erreur lors de l'ajout du commentaire: {str(e)}")
        return jsonify({"success": False, "error": "Erreur serveur"}), 500

from bson.objectid import ObjectId
@app.route('/favoris', methods=['GET', 'POST'])
def favoris():
    if 'user' not in session:
        flash("Veuillez vous connecter", "error")
        return redirect(url_for('compte'))

    user_id = session['user']
    
    # Gestion de la suppression si méthode POST
    if request.method == 'POST':
        interaction_id = request.form.get('interaction_id')
        try:
            # Récupérer l'interaction pour obtenir le place_id
            interaction = interactions_collection.find_one(
                {"_id": ObjectId(interaction_id), "user_id": user_id}
            )
            
            if interaction:
                place_id = interaction['place_id']
                
                # Supprimer l'interaction
                interactions_collection.delete_one({"_id": ObjectId(interaction_id)})
                
                # Retirer le lieu de visited_places
                users_collection.update_one(
                    {"_id": user_id},
                    {"$pull": {"visited_places": place_id}}
                )
                
                flash("Interaction supprimée avec succès", "success")
            else:
                flash("Interaction non trouvée", "error")
        except Exception as e:
            flash(f"Erreur lors de la suppression: {str(e)}", "error")
        return redirect(url_for('favoris'))

    # Méthode GET - Affichage normal
    interactions = list(interactions_collection.find({"user_id": user_id}))
    place_ids = [interaction['place_id'] for interaction in interactions]
    
    lieux = list(lieux_collection.find({"_id": {"$in": place_ids}})) if place_ids else []
    
    lieux_data = []
    for lieu in lieux:
        interaction = next((i for i in interactions if i['place_id'] == lieu['_id']), None)
        
        # Gestion plus robuste des images
        image_drive_url = url_for('static', filename='images/placeholder.jpg')  # Valeur par défaut
        
        if 'images' in lieu and lieu['images']:  # Vérifie si le champ existe et n'est pas None/empty
            try:
                # Si c'est une liste, prendre la première image
                if isinstance(lieu['images'], list):
                    image_url = lieu['images'][0] if lieu['images'] else None
                else:
                    image_url = lieu['images']
                
                # Si on a une URL valide avec 'id='
                if image_url and isinstance(image_url, str) and 'id=' in image_url:
                    image_drive_url = url_for('drive_image', file_id=image_url.split('id=')[-1])
            except Exception as e:
                print(f"Erreur traitement image pour lieu {lieu['_id']}: {str(e)}")
        
        lieux_data.append({
            "lieu_id": lieu['_id'],
            "nom": lieu.get('name', 'Lieu inconnu'),
            "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
            "commune": lieu.get('commune', 'Commune inconnue'),
            "category": lieu.get('category', 'Non catégorisé'),
            "subcategory": lieu.get('subcategory', ''),
            "entry_fee": lieu.get('entry_fee', 'Gratuit'),
            "address": lieu.get('address', ''),
            "image_drive_url": image_drive_url,
            "rating": interaction.get('rating', 0),
            "review": interaction.get('review', ''),
            "interaction_id": str(interaction['_id']) if interaction else None
        })
    
    return render_template('favoris.html', 
                         lieux_data=lieux_data,
                         user={"prenom": session.get('user', 'Utilisateur')})
from flask import jsonify

@app.route('/supprimer_interaction', methods=['POST'])
def supprimer_interaction():
    if 'user' not in session:
        return jsonify(success=False, error="Utilisateur non connecté"), 401

    data = request.get_json()
    interaction_id = data.get('interaction_id')
    user_id = session['user']

    if not interaction_id:
        return jsonify(success=False, error="ID de l'interaction manquant")

    try:
        interaction = interactions_collection.find_one({
            "_id": ObjectId(interaction_id),
            "user_id": user_id
        })

        if not interaction:
            return jsonify(success=False, error="Interaction non trouvée")

        place_id = interaction['place_id']

        # Supprimer l'interaction
        interactions_collection.delete_one({"_id": ObjectId(interaction_id)})

        # Retirer le lieu de visited_places
        users_collection.update_one(
            {"_id": user_id},
            {"$pull": {"visited_places": place_id}}
        )

        return jsonify(success=True)
    
    except Exception as e:
        return jsonify(success=False, error=str(e))

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if 'user' not in session:
        return redirect(url_for('compte'))

    # Récupérer les préférences existantes
    prefs = user_preferences_collection.find_one({"user_id": session['user']}) or {}
    
    if request.method == 'POST':
        try:
            # Récupérer toutes les données du formulaire
            form_data = request.form.to_dict()
            
            # Vérifier qu'au moins un curseur a été modifié (valeur > 0)
            has_adjusted = any(
                float(v) > 0 
                for k, v in form_data.items() 
                if v.replace('.', '').isdigit() and 0 <= float(v) <= 1
            )
            
            if not has_adjusted:
                flash("Vous devez ajuster au moins un curseur avant de soumettre", "error")
                return redirect(url_for('preferences'))
            
            # Préparer les données pour l'enregistrement
            preferences_data = {
                k: float(v) 
                for k, v in form_data.items() 
                if v.replace('.', '').isdigit() and 0 <= float(v) <= 1
            }
            
            # Enregistrer les préférences
            user_preferences_collection.update_one(
                {"user_id": session['user']},
                {"$set": {
                    "preferences": preferences_data,
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            return redirect(url_for('accueil'))
            
        except Exception as e:
            flash(f"Une erreur est survenue: {str(e)}", "error")
            return redirect(url_for('preferences'))

    # GET - Préparer les données pour le template (inchangé)
    all_categories = [
        'Naturel', 'Historique', 'Culturels', 
        'shopping', 'Loisir et Divertissement', 'Religieux',
        'Architectural, artificiel et patrimonial bâti', 
        'Bien-être / Thérapeutique', 'artistiques'
    ]
    
    all_subcategories = [
        'Tour', 'Eglise', 'Site Archeologique', 'Monument', 'parc national',
        'foret', 'Montagne', 'Plage', 'Centre commercial', 'Spa', 'Zoo',
        'Parc d\'attraction', 'Pont', 'musée', 'cinéma', 'Galerie d\'art',
        'theatre', 'Parc aquatique', 'Parc urbain', 'Village', 'Cascades',
        'grotte', 'Lac', 'Palais', 'Port', 'Île', 'parc naturel', 'péninsule',
        'Barrage', 'phare', 'ville', 'Oasis', 'Activités', 'gorges',
        'place urbaine', 'rivière', 'Dunes', 'Tunnel', 'château', 'Jardin',
        'Piscine', 'Stade', 'Marais', 'Fort'
    ]
    
    existing_prefs = prefs.get('preferences', {})
    
    return render_template('preferences.html',
                         all_categories=all_categories,
                         all_subcategories=all_subcategories,
                         existing_prefs=existing_prefs)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if 'user' not in session:
        flash("Veuillez vous connecter", "error")
        return redirect(url_for('compte'))

    user_id = session['user']
    
    # Check if user has interactions
    has_interactions = interactions_collection.count_documents({"user_id": user_id}) > 0
    
    if has_interactions:
        try:
            selected_saison = request.args.get('saison')
            selected_wilaya = request.args.get('wilaya')
            
            if selected_saison or selected_wilaya:
                all_recommendations = dl_recommender.generate_recommendations(
                    wilaya_name=selected_wilaya,
                    season=selected_saison
                )
                recommended_places = all_recommendations.get("recommendations", {}).get(str(user_id), {}).get("recommendations", [])
                manual_selection = True
            else:
                current_wilaya = None
                try:
                    g = geocoder.ip('me')
                    if g.latlng:
                        geolocator = Nominatim(user_agent="bleditrip_app")
                        location = geolocator.reverse(g.latlng, language='fr')
                        if location and 'address' in location.raw:
                            address = location.raw['address']
                            current_wilaya = address.get('state') or address.get('county') or address.get('region')
                except Exception as e:
                    print(f"Erreur géolocalisation: {str(e)}")
                
                all_recommendations = dl_recommender.generate_recommendations(
                    wilaya_name=current_wilaya,
                    season=dl_recommender.get_current_season()
                )
                recommended_places = all_recommendations.get("recommendations", {}).get(str(user_id), {}).get("recommendations", [])
                manual_selection = False
            
            wilaya = None
            try:
                g = geocoder.ip('me')
                if g.latlng:
                    geolocator = Nominatim(user_agent="bleditrip_app")
                    location = geolocator.reverse(g.latlng, language='fr')
                    if location and 'address' in location.raw:
                        address = location.raw['address']
                        wilaya = address.get('state') or address.get('county') or address.get('region')
            except Exception as e:
                print(f"Erreur géolocalisation: {str(e)}")
            
            saison_actuelle = dl_recommender.get_current_season()
            wilayas = list(lieux_collection.distinct("wilaya"))
            wilayas.sort()
            
            lieux_data = []
            for place in recommended_places:
                image_drive_url = url_for('static', filename='images/placeholder.jpg')
                if place['images']:
                    try:
                        image_url = place['images'][0] if isinstance(place['images'], list) else place['images']
                        if isinstance(image_url, str) and 'id=' in image_url:
                            image_id = image_url.split('id=')[-1]
                            image_drive_url = url_for('drive_image', file_id=image_id)
                    except Exception as e:
                        print(f"Error processing image URL: {str(e)}")
                
                lieux_data.append({
                    "lieu_id": place['place_id'],
                    "name": place['name'],
                    "wilaya": place['wilaya'],
                    "commune": place['commune'],
                    "category": place['category'],
                    "subcategory": place['subcategory'],
                    "entry_fee": place['entry_fee'],
                    "address": place['address'],
                    "image_drive_url": image_drive_url,
                    "model_type": "deep_learning"
                })
            
            return render_template('recommendations.html',
                                user={"prenom": session.get('user', 'Utilisateur')},
                                lieux_data=lieux_data,
                                wilaya=wilaya,
                                saison=saison_actuelle,
                                wilayas=wilayas,
                                saisons=["Printemps", "Eté", "Automne", "Hiver"],
                                manual_selection=manual_selection,
                                recommender_type="deep_learning")
                    
        except Exception as e:
            print(f"Erreur avec le modèle deep learning: {str(e)}")
            
            # Retourner une liste vide pour afficher le message "Aucune recommandation"
            return render_template('recommendations.html',
                                user={"prenom": session.get('user', 'Utilisateur')},
                                lieux_data=[],
                                wilaya=None,
                                saison=dl_recommender.get_current_season(),
                                wilayas=list(lieux_collection.distinct("wilaya")),
                                saisons=["Printemps", "Eté", "Automne", "Hiver"],
                                manual_selection=False,
                                recommender_type="deep_learning")
    else:
        has_preferences = user_preferences_collection.count_documents({"user_id": user_id}) > 0
        
        if not has_preferences:
            # Au lieu de rediriger vers les préférences, on affiche la page avec une liste vide
            return render_template('recommendations.html',
                                user={"prenom": session.get('user', 'Utilisateur')},
                                lieux_data=[],
                                wilaya=None,
                                saison=dl_recommender.get_current_season(),
                                wilayas=list(lieux_collection.distinct("wilaya")),
                                saisons=["Printemps", "Eté", "Automne", "Hiver"],
                                manual_selection=False,
                                recommender_type="no_preferences")
            
        try:
            selected_saison = request.args.get('saison')
            selected_wilaya = request.args.get('wilaya')
            
            if selected_saison or selected_wilaya:
                recommended_places = recommender.recommend_places_with_params(
                    user_id, 
                    saison=selected_saison,
                    wilaya=selected_wilaya
                )
                manual_selection = True
            else:
                recommended_places = recommender.recommend_places_with_params(user_id)
                manual_selection = False
            
            wilaya = None
            try:
                g = geocoder.ip('me')
                if g.latlng:
                    geolocator = Nominatim(user_agent="bleditrip_app")
                    location = geolocator.reverse(g.latlng, language='fr')
                    if location and 'address' in location.raw:
                        address = location.raw['address']
                        wilaya = address.get('state') or address.get('county') or address.get('region')
            except Exception as e:
                print(f"Erreur géolocalisation: {str(e)}")
            
            saison_actuelle = recommender.get_current_season()
            wilayas = list(lieux_collection.distinct("wilaya"))
            wilayas.sort()
            
            lieux_data = []
            for place in recommended_places:
                image_drive_url = url_for('static', filename='images/placeholder.jpg')
                if 'images' in place and place['images']:
                    try:
                        image_url = place['images']
                        if isinstance(image_url, str) and 'id=' in image_url:
                            image_id = image_url.split('id=')[-1]
                            image_drive_url = url_for('drive_image', file_id=image_id)
                    except Exception as e:
                        print(f"Error processing image URL: {str(e)}")
                
                lieux_data.append({
                    "lieu_id": place['_id'],
                    "name": place.get('name', 'Lieu inconnu'),
                    "wilaya": place.get('wilaya', 'Wilaya inconnue'),
                    "commune": place.get('commune', 'Commune inconnue'),
                    "category": place.get('category', 'Non catégorisé'),
                    "subcategory": place.get('subcategory', ''),
                    "entry_fee": place.get('entry_fee', ''),
                    "address": place.get('address', ''),
                    "image_drive_url": image_drive_url,
                    "model_type": "content_based"
                })
            
            return render_template('recommendations.html',
                                user={"prenom": session.get('user', 'Utilisateur')},
                                lieux_data=lieux_data,
                                wilaya=wilaya,
                                saison=saison_actuelle,
                                wilayas=wilayas,
                                saisons=["Printemps", "Eté", "Automne", "Hiver"],
                                manual_selection=manual_selection,
                                recommender_type="content_based")
            
        except Exception as e:
            print(f"Erreur recommandation content-based: {str(e)}")
            return render_template('recommendations.html',
                                user={"prenom": session.get('user', 'Utilisateur')},
                                lieux_data=[],
                                wilaya=None,
                                saison=dl_recommender.get_current_season(),
                                wilayas=list(lieux_collection.distinct("wilaya")),
                                saisons=["Printemps", "Eté", "Automne", "Hiver"],
                                manual_selection=False,
                                recommender_type="error")

@app.route('/recommendations2', methods=['GET', 'POST'])
def recommendations2():
    if 'user' not in session:
        flash("Veuillez vous connecter", "error")
        return redirect(url_for('compte'))

    user_id = session['user']
    
    # Vérifier le type d'utilisateur
    has_interactions = interactions_collection.count_documents({"user_id": user_id}) > 0
    has_preferences = user_preferences_collection.count_documents({"user_id": user_id}) > 0
    
    # Récupérer les paramètres de filtrage
    selected_saison = request.args.get('saison')
    selected_wilaya = request.args.get('wilaya')
    manual_selection = bool(selected_saison or selected_wilaya)
    
    try:
        # Cas 1: Utilisateur avec interactions (DL Recommender)
        if has_interactions:
            recommender_to_use = dl_recommender2
            model_type = "deep_learning"
            
            if manual_selection:
                all_recommendations = dl_recommender2.generate_recommendations(
                    wilaya_name=selected_wilaya,
                    season=selected_saison
                )
            else:
                # Géolocalisation automatique
                current_wilaya = None
                try:
                    g = geocoder.ip('me')
                    if g.latlng:
                        geolocator = Nominatim(user_agent="bleditrip_app")
                        location = geolocator.reverse(g.latlng, language='fr')
                        if location and 'address' in location.raw:
                            address = location.raw['address']
                            current_wilaya = address.get('state') or address.get('county') or address.get('region')
                except Exception as e:
                    print(f"Erreur géolocalisation: {str(e)}")
                
                all_recommendations = dl_recommender2.generate_recommendations(
                    wilaya_name=current_wilaya,
                    season=dl_recommender2.get_current_season()
                )
            
            recommended_places = all_recommendations.get("recommendations", {}).get(str(user_id), {}).get("recommendations", [])
        
        # Cas 2: Utilisateur avec préférences mais sans interactions (Content-Based)
        elif has_preferences:
            recommender_to_use = recommender
            model_type = "content_based"
            
            if manual_selection:
                recommended_places = recommender.recommend_places_with_params(
                    user_id, 
                    saison=selected_saison,
                    wilaya=selected_wilaya
                )
            else:
                recommended_places = recommender.recommend_places_with_params(user_id)
            
            # Ajouter les ratings moyens
            for place in recommended_places:
                place['predicted_rating'] = get_average_rating(place['_id'])
        
        # Cas 3: Nouvel utilisateur (Recommandations générales)
        else:
            model_type = "general"
            recommended_places = []
            
            # Récupérer les lieux les mieux notés
            pipeline = [
                {"$match": {"wilaya": selected_wilaya} if selected_wilaya else {}},
                {"$sample": {"size": 20}},
                {"$project": {
                    "_id": 1,
                    "name": 1,
                    "wilaya": 1,
                    "commune": 1,
                    "category": 1,
                    "subcategory": 1,
                    "entry_fee": 1,
                    "address": 1,
                    "images": 1,
                    "average_rating": 1
                }}
            ]
            
            general_recommendations = list(lieux_collection.aggregate(pipeline))
            
            for place in general_recommendations:
                recommended_places.append({
                    "place_id": place["_id"],
                    "name": place.get("name", "Lieu inconnu"),
                    "wilaya": place.get("wilaya", "Wilaya inconnue"),
                    "commune": place.get("commune", "Commune inconnue"),
                    "category": place.get("category", "Non catégorisé"),
                    "subcategory": place.get("subcategory", ""),
                    "entry_fee": place.get("entry_fee", ""),
                    "address": place.get("address", ""),
                    "images": place.get("images", []),
                    "predicted_rating": place.get("average_rating", 3.0)
                })
        
        # Géolocalisation pour l'affichage
        wilaya = None
        try:
            g = geocoder.ip('me')
            if g.latlng:
                geolocator = Nominatim(user_agent="bleditrip_app")
                location = geolocator.reverse(g.latlng, language='fr')
                if location and 'address' in location.raw:
                    address = location.raw['address']
                    wilaya = address.get('state') or address.get('county') or address.get('region')
        except Exception as e:
            print(f"Erreur géolocalisation: {str(e)}")
        
        # Préparation des données pour le template
        saison_actuelle = dl_recommender2.get_current_season() if has_interactions else recommender.get_current_season()
        wilayas = list(lieux_collection.distinct("wilaya"))
        wilayas.sort()
        
        lieux_data = []
        for place in recommended_places:
            image_drive_url = url_for('static', filename='images/placeholder.jpg')
            images = place.get('images', [])
            
            if images:
                try:
                    image_url = images[0] if isinstance(images, list) else images
                    if isinstance(image_url, str) and 'id=' in image_url:
                        image_id = image_url.split('id=')[-1]
                        image_drive_url = url_for('drive_image', file_id=image_id)
                except Exception as e:
                    print(f"Error processing image URL: {str(e)}")
            
            lieux_data.append({
                "lieu_id": place.get('place_id', place.get('_id')),
                "name": place.get('name', 'Lieu inconnu'),
                "wilaya": place.get('wilaya', 'Wilaya inconnue'),
                "commune": place.get('commune', 'Commune inconnue'),
                "category": place.get('category', 'Non catégorisé'),
                "subcategory": place.get('subcategory', ''),
                "entry_fee": place.get('entry_fee', ''),
                "address": place.get('address', ''),
                "image_drive_url": image_drive_url,
                "predicted_rating": place.get('predicted_rating', 3.0),
                "model_type": model_type
            })
        
        return render_template('recommendations2.html',
                            user={"prenom": session.get('user', 'Utilisateur')},
                            lieux_data=lieux_data,
                            wilaya=wilaya,
                            saison=saison_actuelle,
                            wilayas=wilayas,
                            saisons=["Printemps", "Eté", "Automne", "Hiver"],
                            manual_selection=manual_selection,
                            recommender_type=model_type)
    
    except Exception as e:
        print(f"Erreur avec le système de recommandation: {str(e)}")
        return render_template('recommendations2.html',
                            user={"prenom": session.get('user', 'Utilisateur')},
                            lieux_data=[],
                            wilaya=None,
                            saison=dl_recommender2.get_current_season(),
                            wilayas=list(lieux_collection.distinct("wilaya")),
                            saisons=["Printemps", "Eté", "Automne", "Hiver"],
                            manual_selection=False,
                            recommender_type="error")
# Import TimeMatrixCalculator (make sure the module exists in your project)
from time_matrix_calculator import TimeMatrixCalculator  # Adjust the import path as needed

# Initialize TimeMatrixCalculator with OpenRoute API key
api_key = os.environ.get('5b3ce3597851110001cf6248624a23ad34214d319e9e4a33ec00eebb')
time_matrix_calculator = TimeMatrixCalculator(api_key)

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
        user_id = session['user']
        selected_place_ids = [int(pid) for pid in session['selected_places']]
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
        
        # Vérifier si l'utilisateur a des interactions
        has_interactions = interactions_collection.count_documents({"user_id": user_id}) > 0
        
        # Récupérer les ratings selon le type d'utilisateur
        if has_interactions:
            # Cas utilisateur avec interactions (DL Recommender)
            all_recommendations = dl_recommender2.generate_recommendations(
                wilaya_name=None,
                season=dl_recommender2.get_current_season()
            )
            user_recommendations = all_recommendations.get("recommendations", {}).get(str(user_id), {})
            recommended_places = user_recommendations.get("recommendations", [])
            
            # Mapping des ratings prédits
            predicted_ratings = {
                int(p['_id']): p.get('predicted_rating', 3.0)
                for p in recommended_places
                if '_id' in p
            }
        else:
            # Cas utilisateur sans interactions (Content-Based)
            recommended_places = recommender.recommend_places_with_params(user_id)
            
            # Utiliser les ratings moyens réels comme fallback
            predicted_ratings = {}
            for place in recommended_places:
                place_id = place['_id']
                predicted_ratings[place_id] = get_average_rating(place_id)
            
        places = []
        for place_id in selected_place_ids:
            place = lieux_collection.find_one({"_id": place_id})
            if place and 'coordinates' in place:
                # Utiliser le rating approprié selon le cas
                rating = predicted_ratings.get(place_id, 3.0)
                
                places.append({
                    '_id': place_id,
                    'name': place.get('name', 'Unknown Place'),
                    'coordinates': place['coordinates'],
                    'category': place.get('category', ''),
                    'visit_time_minutes': float(place.get('average_visit_time', 1.0)) * 60,
                    'visit_time_hours': place.get('average_visit_time', 1.0),
                    'rating': rating  # Rating prédit ou moyen selon le cas
                })
        
        if len(places) < 2:
            flash("Not enough valid places", "error")
            return redirect(url_for('recommendations2'))
        
        # Use the TimeMatrixCalculator to get the time matrix
        calculator = TimeMatrixCalculator(api_key)
        
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
        visit_time = place['visit_time_minutes']
        
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
            'rating': place['rating'], 
            'start': visit_start.strftime('%H:%M'),
            'end': visit_end.strftime('%H:%M'),
            'duration': visit_time,
            'display_duration': f"{place['visit_time_hours']}h",
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

def get_average_rating(place_id):
    """Calcule la note moyenne d'un lieu depuis la collection interactions"""
    ratings = list(interactions_collection.find(
        {"place_id": place_id}, 
        {"rating": 1}
    ))
    
    if not ratings:
        # Fallback 1: Rating moyen par catégorie
        place = lieux_collection.find_one({"_id": place_id})
        if place and 'category' in place:
            category_avg = lieux_collection.aggregate([
                {"$match": {"category": place['category']}},
                {"$group": {"_id": None, "avgRating": {"$avg": "$average_rating"}}}
            ])
            if category_avg:
                return round(category_avg[0]['avgRating'], 1)
        
        # Fallback 2: Valeur par défaut
        return 3.0
    
    total = sum(r['rating'] for r in ratings)
    return round(total / len(ratings), 1)

@app.route('/sansauthentification')
def sansauthentification():
    """Page à propos accessible sans authentification"""
    # Récupérer toutes les wilayas disponibles
    wilayas = lieux_collection.distinct("wilaya")
    wilayas.sort()  # Trier par ordre alphabétique
    
    # Récupérer la wilaya sélectionnée (soit par paramètre, soit par géolocalisation)
    selected_wilaya = request.args.get('wilaya')
    
    try:
        g = geocoder.ip('me')
        latitude, longitude = g.latlng
        
        # Reverse geocoding pour obtenir la wilaya
        geolocator = Nominatim(user_agent="bleditrip_app")
        location = geolocator.reverse((latitude, longitude), language='fr')
        
        detected_wilaya = None
        commune = None
        if location and 'address' in location.raw:
            address = location.raw['address']
            detected_wilaya = address.get('state') or address.get('county') or address.get('region')
            commune = address.get('town') or address.get('village') or address.get('city')
        
        # Déterminer la wilaya à utiliser (priorité au filtre sélectionné)
        current_wilaya = selected_wilaya or detected_wilaya
        
        # Récupérer les lieux selon la wilaya choisie
        lieux_data = []
        if current_wilaya:
            query = {"wilaya": {"$regex": f"^{current_wilaya}$", "$options": "i"}}
            lieux = lieux_collection.find(query)
            
            for lieu in lieux:
                # Initialisation avec l'image par défaut
                image_drive_url = url_for('static', filename='images/placeholder.jpg')
                # Gestion sécurisée des images comme dans recommended_places
                if 'images' in lieu and lieu['images']:
                    try:
                        # Gère à la fois les chaînes et les listes d'images
                        image_url = lieu['images'][0] if isinstance(lieu['images'], list) else lieu['images']
                        
                        if isinstance(image_url, str) and 'id=' in image_url:
                            image_id = image_url.split('id=')[-1].split('&')[0]  # Plus sécurisé
                            image_drive_url = url_for('drive_image', file_id=image_id)
                    except Exception as e:
                        print(f"Erreur image pour lieu {lieu.get('_id', 'inconnu')}: {str(e)}")
                        # L'image par défaut est déjà définie

                # Le reste du code reste inchangé
                lieux_data.append({
                    "lieu_id": str(lieu['_id']),
                    "nom": lieu.get('name', 'Lieu inconnu'),
                    "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
                    "commune": lieu.get('commune', 'Commune inconnue'),
                    "category": lieu.get('category', 'Non catégorisé'),
                    "subcategory": lieu.get('subcategory', ''),
                    "entry_fee": lieu.get('entry_fee', ""),
                    "address": lieu.get('address', ""),
                    "image_drive_url": image_drive_url,
                    "position": (latitude, longitude),
                    "wilaya_detectee": detected_wilaya
                })
        
        return render_template('sansauthentification.html', 
                             lieux_data=lieux_data,
                             position=(latitude, longitude),
                             wilaya=current_wilaya,
                             wilayas=wilayas,
                             current_wilaya=current_wilaya)
    
    except Exception as e:
        print(f"Erreur de géolocalisation: {str(e)}")
        # Fallback sans géolocalisation
        current_wilaya = selected_wilaya
        lieux_data = []
        
        query = {}
        if current_wilaya:
            query = {"wilaya": {"$regex": f"^{current_wilaya}$", "$options": "i"}}
        
        lieux = lieux_collection.find(query).limit(20)
        
        for lieu in lieux:
            if 'images' in lieu:
                image_url = lieu['images']
                image_id = image_url.split('id=')[-1]
                image_drive_url = url_for('drive_image', file_id=image_id)
            else:
                image_drive_url = url_for('static', filename='images/placeholder.jpg')

            lieux_data.append({
                "lieu_id": str(lieu['_id']),
                "nom": lieu.get('name', 'Lieu inconnu'),
                "wilaya": lieu.get('wilaya', 'Wilaya inconnue'),
                "commune": lieu.get('commune', 'Commune inconnue'),
                "category": lieu.get('category', 'Non catégorisé'),
                "subcategory": lieu.get('subcategory', ''),
                "entry_fee": lieu.get('entry_fee', ""),
                "address": lieu.get('address', ""),
                "image_drive_url": image_drive_url,
                "wilaya_detectee": None
            })
        
        return render_template('sansauthentification.html',
                             lieux_data=lieux_data,
                             position=None,
                             wilaya=current_wilaya,
                             wilayas=wilayas,
                             current_wilaya=current_wilaya)

if __name__ == "__main__":
    app.run(debug=True)