import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torchdiffeq import odeint_adjoint as odeint
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
import json
import geocoder
from datetime import datetime, date
import pickle
import unicodedata
from tqdm import tqdm
from geopy.geocoders import Nominatim
import numpy as np 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings

class AlgeriaTourismRecommender:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="Algeria", model_path="tourism_recommender_complete2.pkl"):
        # Configuration initiale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.model_path = model_path
        
        # Initialisation des composants
        self.model = None
        self.saved_data = None
        self.graph = None
        self.load_model()
        
        # Paramètres actuels
        self.current_wilaya = None
        self.current_season = None
    
    # ==================== PARTIE MODÈLE ====================
    
    class ODEBlock(nn.Module):
        def __init__(self, odefunc, method='dopri5', rtol=1e-4, atol=1e-5):
            super().__init__()
            self.odefunc = odefunc
            self.method = method
            self.rtol = rtol
            self.atol = atol
            
        def forward(self, x):
            integration_time = torch.tensor([0, 1]).float().to(x.device)
            out = odeint(self.odefunc, x, integration_time, 
                        method=self.method, rtol=self.rtol, atol=self.atol)
            return out[1]

    class EnhancedMLPEncoder(nn.Module):
        def __init__(self, input_dim, latent_dim=64, dropout=0.2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, latent_dim),
                nn.Tanh()
            )
            
        def forward(self, x, training=False):
            if not training:
                self.encoder[1].eval()
            return self.encoder(x), None

    class EnhancedODEFunc(nn.Module):
        def __init__(self, latent_dim, edge_index, edge_attr):
            super().__init__()
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            
            self.diffusion = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh()
            )
            
            self.attention = nn.Sequential(
                nn.Linear(latent_dim * 2 + 1, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, 1)
            )
            
            src = edge_index[0]
            self.degree = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
            self.degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
            self.degree = torch.clamp(self.degree, min=1)
            
        def forward(self, t, x):
            local_diff = self.diffusion(x)
            src, dst = self.edge_index[0], self.edge_index[1]
            src_emb, dst_emb = x[src], x[dst]
            edge_attr = self.edge_attr

            concat = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
            attention_weights = torch.sigmoid(self.attention(concat))
            messages = attention_weights * local_diff[src]

            out = torch.zeros_like(x)
            out = out.index_add(0, dst, messages)
            out = out / self.degree.unsqueeze(1)

            return out

    class EnhancedGODEModel(nn.Module):
        def __init__(self, num_users, num_items, latent_dim, graph):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.latent_dim = latent_dim
            
            self.odeblock = AlgeriaTourismRecommender.ODEBlock(
                AlgeriaTourismRecommender.EnhancedODEFunc(latent_dim, graph.edge_index, graph.edge_attr),
                method='dopri5', rtol=1e-4, atol=1e-5)
            
            self.predictor = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x, user_idx, item_idx):
            evolved_emb = self.odeblock(x)
            users, items = evolved_emb[:self.num_users], evolved_emb[self.num_users:]
            
            user_emb = users[user_idx]
            item_emb = items[item_idx]
            
            combined = torch.cat([user_emb, item_emb], dim=1)
            pred = self.predictor(combined)
            return (pred * 4 + 1).squeeze()

    # ==================== FONCTIONS UTILITAIRES ====================

    @staticmethod
    def normalize_string(s):
        if pd.isna(s):
            return ""
        return unicodedata.normalize('NFKD', str(s).lower()).encode('ascii', 'ignore').decode('utf-8').strip()

    def load_model(self):
        """Charge le modèle sauvegardé avec tous ses composants"""
        with open(self.model_path, 'rb') as f:
            self.saved_data = pickle.load(f)
        
        # Reconstruire le graphe
        graph_data = self.saved_data['graph_data']
        self.graph = Data(
            x=torch.tensor(graph_data['x']).clone().detach().to(self.device),
            edge_index=torch.tensor(graph_data['edge_index']).clone().detach().to(self.device),
            edge_attr=torch.tensor(graph_data['edge_attr']).clone().detach().to(self.device)
        )
        
        # Reconstruire le modèle
        self.model = self.EnhancedGODEModel(
            num_users=self.saved_data['model_config']['num_users'],
            num_items=self.saved_data['model_config']['num_items'],
            latent_dim=self.saved_data['model_config']['latent_dim'],
            graph=self.graph
        ).to(self.device)
        
        self.model.load_state_dict(self.saved_data['model_state_dict'])
        self.model.eval()

    def detect_wilaya(self):
        """Détecte la wilaya actuelle en utilisant la géolocalisation"""
        try:
            print("Détection de la position...")
            g = geocoder.ip('me')

            if not g.ok or not g.latlng:
                print("Échec de la détection via l'adresse IP.")
                return None

            latitude, longitude = g.latlng
            print(f"Coordonnées détectées : {latitude}, {longitude}")

            geolocator = Nominatim(user_agent="algeria_tourism_app")
            location = geolocator.reverse((latitude, longitude), language='fr')

            if location and 'address' in location.raw:
                address = location.raw['address']
                wilaya = address.get('state') or address.get('county') or address.get('region')
                if wilaya:
                    print("Wilaya détectée :", wilaya)
                    self.current_wilaya = wilaya.strip()
                    return self.current_wilaya
                else:
                    print("Aucune wilaya trouvée dans l'adresse.")
            else:
                print("Adresse introuvable pour ces coordonnées.")
        except Exception as e:
            print("Erreur de géolocalisation :", e)

        self.current_wilaya = "Alger"
        return self.current_wilaya

    def get_current_season(self):
        """Détermine la saison actuelle basée sur la date"""
        today = date.today()
        y = today.year

        spring = (date(y, 3, 21), date(y, 6, 20))
        summer = (date(y, 6, 21), date(y, 9, 22))
        autumn = (date(y, 9, 23), date(y, 12, 20))
        
        if spring[0] <= today <= spring[1]:
            self.current_season = "Printemps"
        elif summer[0] <= today <= summer[1]:
            self.current_season = "Eté"
        elif autumn[0] <= today <= autumn[1]:
            self.current_season = "Automne"
        else:
            self.current_season = "Hiver"
            
        return self.current_season

    def prepare_items_data(self, items_df):
        """Prépare les données des lieux pour la prédiction"""
        place_id_encoder = self.saved_data['place_id_encoder']
        items_df['place_id_enc'] = place_id_encoder.transform(items_df['_id'])
        return items_df
    
    def get_encoded_user_id(self, user_id):
        """Get the encoded user ID for the model"""
        try:
            if str(user_id) in self.saved_data['user_id_encoder'].classes_:
                return self.saved_data['user_id_encoder'].transform([str(user_id)])[0]
            return None
        except Exception as e:
            print(f"Error encoding user ID {user_id}: {str(e)}")
            return None
    
    def create_user_features(self, user_id):
        """Create feature vector for new users based on their preferences or basic profile"""
        try:
            # Check if user exists
            user_doc = self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return None
                
            # Try to get user preferences
            user_prefs = self.db.user_preferences.find_one({"user_id": user_id})
            
            # Create a basic feature vector
            feature_vector = np.zeros(self.saved_data['user_features'].shape[1])
            
            # Fill with basic demographic info if available
            if 'age' in user_doc and user_doc['age']:
                # Normalize age to 0-1 range assuming model trained on ages 10-100
                norm_age = (max(10, min(100, user_doc['age'])) - 10) / 90
                feature_vector[0] = norm_age
                
            if 'sexe' in user_doc:
                # Encode sex (assuming same encoding as training: F=1, M=0)
                feature_vector[1] = 1 if user_doc['sexe'].upper() == 'F' else 0
            
            # If we have preferences, use them to create a more detailed feature vector
            if user_prefs and 'preferences' in user_prefs:
                prefs = user_prefs['preferences']
                
                strongest_prefs = sorted(
                    [(k, v) for k, v in prefs.items() if isinstance(v, (int, float))],
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]  # Top 5 preferences
                
                for i, (pref_name, pref_value) in enumerate(strongest_prefs):
                    if i + 2 < len(feature_vector):
                        feature_vector[i + 2] = min(1.0, float(pref_value))
            
            # Apply the same scaling as during training
            if 'user_scaler' in self.saved_data:
                # Convert to DataFrame with feature names if available
                if hasattr(self.saved_data['user_scaler'], 'feature_names_in_'):
                    feature_names = self.saved_data['user_scaler'].feature_names_in_
                    df = pd.DataFrame([feature_vector], columns=feature_names)
                    feature_vector = self.saved_data['user_scaler'].transform(df).flatten()
                else:
                    feature_vector = self.saved_data['user_scaler'].transform(
                        feature_vector.reshape(1, -1)).flatten()
                
            return feature_vector
            
        except Exception as e:
            print(f"Error creating user features: {str(e)}")
            return None
    
    def find_similar_users(self, user_features, top_k=5):
        """Find the most similar users in the training set"""
        try:
            if self.saved_data['user_features'].shape[0] == 0:
                return np.array([])  # Return empty array instead of None
                
            similarities = cosine_similarity(user_features.reshape(1, -1), 
                                        self.saved_data['user_features'])
            
            # Get top-k similar users (ensure we return array even if empty)
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            return top_indices
            
        except Exception as e:
            print(f"Error finding similar users: {str(e)}")
            return np.array([])  # Return empty array on error


    def generate_recommendations(self, wilaya_name=None, season=None, min_rating=3.0):
        """
        Génère des recommandations pour tous les utilisateurs (existants et nouveaux)
        
        Args:
            wilaya_name (str): Nom de la wilaya pour filtrer les lieux
            season (str): Saison pour filtrer les lieux
            min_rating (float): Seuil minimal de note pour les recommandations
            
        Returns:
            dict: Dictionnaire contenant les recommandations pour tous les utilisateurs
        """
        # Détermination de la wilaya et saison
        self.current_wilaya = wilaya_name or self.detect_wilaya() or "Alger"
        self.current_season = season or self.get_current_season()
        
        print(f"\nGénération des recommandations pour {self.current_wilaya} en {self.current_season}...")
        
        # Requête pour les lieux
        query = {'wilaya': self.current_wilaya}
        if self.current_season:
            season_regex = self.current_season.lower()
            query['best_season'] = {'$regex': season_regex, '$options': 'i'}
        
        # Récupération des lieux
        filtered_items = list(self.db.lieux.find(query))
        
        if not filtered_items:
            print("Aucun lieu trouvé avec les critères actuels")
            return None
        
        # Préparation des données
        filtered_items_df = pd.DataFrame(filtered_items)
        filtered_items_df = self.prepare_items_data(filtered_items_df)
        wilaya_indices = torch.tensor(filtered_items_df['place_id_enc'].values, dtype=torch.long).to(self.device)
        
        # Récupération des utilisateurs
        users = list(self.db.users.find({}, {'_id': 1, 'prenom': 1, 'nom': 1, 'age': 1, 'sexe': 1}))
        if not users:
            print("Aucun utilisateur trouvé dans la base")
            return None
        
        # Structure des résultats
        all_recommendations = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "wilaya": self.current_wilaya,
                "season": self.current_season,
                "min_rating": min_rating,
                "total_users": len(users)
            },
            "recommendations": {}
        }
        
        # Traitement des utilisateurs
        for user in tqdm(users, desc="Traitement des utilisateurs"):
            user_id = str(user["_id"])
            user_name = f"{user.get('prenom', '')} {user.get('nom', '')}".strip()
            
            # Vérification utilisateur
            user_id_enc = self.get_encoded_user_id(user_id)
            
            if user_id_enc is not None:
                # Utilisateur existant
                user_tensor = torch.full((len(filtered_items_df),), user_id_enc, dtype=torch.long).to(self.device)
                with torch.no_grad():
                    preds = self.model(self.graph.x, user_tensor, wilaya_indices).cpu().numpy()
            else:
                # Nouvel utilisateur
                user_features = self.create_user_features(user["_id"])
                if user_features is None:
                    continue
                
                similar_users = self.find_similar_users(user_features, top_k=5)
                if len(similar_users) == 0:
                    continue
                
                all_preds = []
                for similar_user_enc in similar_users:
                    user_tensor = torch.full((len(filtered_items_df),), similar_user_enc, dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        preds = self.model(self.graph.x, user_tensor, wilaya_indices).cpu().numpy()
                        all_preds.append(preds)
                
                preds = np.mean(all_preds, axis=0)
            
            # Formatage des résultats
            user_recs = []
            for item, pred in zip(filtered_items, preds):
                if pred >= min_rating:
                    rec = item.copy()
                    rec['_id'] = str(rec['_id'])
                    rec['place_id'] = str(rec['_id'])
                    rec['predicted_rating'] = float(pred)
                    user_recs.append(rec)
            
            user_recs.sort(key=lambda x: x["predicted_rating"], reverse=True)
            
            if user_recs:
                all_recommendations["recommendations"][user_id] = {
                    "user_name": user_name,
                    "user_type": "existing" if user_id_enc is not None else "new",
                    "recommendations": user_recs,
                    "count": len(user_recs)
                }
        
        return all_recommendations
    
warnings.filterwarnings("ignore", category=UserWarning)  # Ignorer les UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignorer les FutureWarnings

