import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torchdiffeq import odeint_adjoint as odeint
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
import geocoder
from datetime import datetime, date
import pickle
import unicodedata
from tqdm import tqdm
from geopy.geocoders import Nominatim
import numpy as np
import warnings

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AlgeriaTourismRecommender1:
    """
    Système de recommandation touristique pour l'Algérie utilisant un modèle 
    Graph-ODE pour prédire les préférences des utilisateurs.
    """
    
    # Constantes de configuration
    DEFAULT_MIN_RATING = 3.0
    MAX_RECOMMENDATIONS = 10
    DEFAULT_WILAYA = "Alger"
    DEFAULT_LATENT_DIM = 64
    
    def __init__(self, mongo_uri="mongodb+srv://Melissa:Melissa@cluster0.dmascbk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0", 
                 db_name="Algeria", model_path="tourism_recommender_complete2.pkl"):
        """
        Initialise le système de recommandation.
        
        Args:
            mongo_uri (str): URI de connexion MongoDB
            db_name (str): Nom de la base de données
            model_path (str): Chemin vers le modèle sauvegardé
        """
        # Configuration matériel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du device: {self.device}")
        
        # Connexion base de données
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.model_path = model_path
        
        # Composants du modèle
        self.model = None
        self.saved_data = None
        self.graph = None
        
        # État actuel
        self.current_wilaya = None
        self.current_season = None
        
        # Chargement du modèle
        self._load_model()
    
    # ==================== MODÈLES NEURAUX ====================
    
    class ODEBlock(nn.Module):
        """Bloc ODE pour l'évolution temporelle des embeddings."""
        
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

    class EnhancedODEFunc(nn.Module):
        """Fonction ODE améliorée avec mécanisme d'attention."""
        
        def __init__(self, latent_dim, edge_index, edge_attr):
            super().__init__()
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            
            # Réseau de diffusion
            self.diffusion = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, latent_dim),
                nn.Tanh()
            )
            
            # Mécanisme d'attention
            self.attention = nn.Sequential(
                nn.Linear(latent_dim * 2 + 1, latent_dim),
                nn.Tanh(),
                nn.Linear(latent_dim, 1)
            )
            
            # Calcul des degrés des nœuds
            src = edge_index[0]
            self.degree = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
            self.degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
            self.degree = torch.clamp(self.degree, min=1)
            
        def forward(self, t, x):
            # Diffusion locale
            local_diff = self.diffusion(x)
            
            # Extraction des arêtes
            src, dst = self.edge_index[0], self.edge_index[1]
            src_emb, dst_emb = x[src], x[dst]
            edge_attr = self.edge_attr

            # Calcul de l'attention
            concat = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
            attention_weights = torch.sigmoid(self.attention(concat))
            messages = attention_weights * local_diff[src]

            # Agrégation des messages
            out = torch.zeros_like(x)
            out = out.index_add(0, dst, messages)
            out = out / self.degree.unsqueeze(1)

            return out

    class EnhancedGODEModel(nn.Module):
        """Modèle Graph-ODE amélioré pour la recommandation."""
        
        def __init__(self, num_users, num_items, latent_dim, graph):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.latent_dim = latent_dim
            
            # Bloc ODE
            self.odeblock = AlgeriaTourismRecommender1.ODEBlock(
                AlgeriaTourismRecommender1.EnhancedODEFunc(
                    latent_dim, graph.edge_index, graph.edge_attr
                ),
                method='dopri5', rtol=1e-4, atol=1e-5
            )
            
            # Prédicteur final
            self.predictor = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x, user_idx, item_idx):
            # Évolution des embeddings via ODE
            evolved_emb = self.odeblock(x)
            users, items = evolved_emb[:self.num_users], evolved_emb[self.num_users:]
            
            # Extraction des embeddings utilisateur et item
            user_emb = users[user_idx]
            item_emb = items[item_idx]
            
            # Prédiction finale
            combined = torch.cat([user_emb, item_emb], dim=1)
            pred = self.predictor(combined)
            return (pred * 4 + 1).squeeze()  # Échelle 1-5

    # ==================== UTILITAIRES ====================
    
    @staticmethod
    def normalize_string(s):
        """Normalise une chaîne de caractères."""
        if pd.isna(s):
            return ""
        return unicodedata.normalize('NFKD', str(s).lower()).encode(
            'ascii', 'ignore').decode('utf-8').strip()

    def _load_model(self):
        """Charge le modèle sauvegardé avec tous ses composants."""
        try:
            with open(self.model_path, 'rb') as f:
                self.saved_data = pickle.load(f)
            
            # Reconstruction du graphe
            graph_data = self.saved_data['graph_data']
            self.graph = Data(
                x=torch.tensor(graph_data['x']).clone().detach().to(self.device),
                edge_index=torch.tensor(graph_data['edge_index']).clone().detach().to(self.device),
                edge_attr=torch.tensor(graph_data['edge_attr']).clone().detach().to(self.device)
            )
            
            # Reconstruction du modèle
            config = self.saved_data['model_config']
            self.model = self.EnhancedGODEModel(
                num_users=config['num_users'],
                num_items=config['num_items'],
                latent_dim=config['latent_dim'],
                graph=self.graph
            ).to(self.device)
            
            self.model.load_state_dict(self.saved_data['model_state_dict'])
            self.model.eval()
            
            print("Modèle chargé avec succès!")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise

    def detect_wilaya(self):
        """Détecte la wilaya actuelle via géolocalisation."""
        try:
            print("Détection de la position géographique...")
            g = geocoder.ip('me')

            if not g.ok or not g.latlng:
                print("Échec de la géolocalisation IP.")
                self.current_wilaya = self.DEFAULT_WILAYA
                return self.current_wilaya

            latitude, longitude = g.latlng
            print(f"Coordonnées détectées: {latitude}, {longitude}")

            geolocator = Nominatim(user_agent="algeria_tourism_recommender")
            location = geolocator.reverse((latitude, longitude), language='fr')

            if location and 'address' in location.raw:
                address = location.raw['address']
                wilaya = (address.get('state') or 
                         address.get('county') or 
                         address.get('region'))
                
                if wilaya:
                    self.current_wilaya = wilaya.strip()
                    print(f"Wilaya détectée: {self.current_wilaya}")
                    return self.current_wilaya

        except Exception as e:
            print(f"Erreur de géolocalisation: {e}")

        self.current_wilaya = self.DEFAULT_WILAYA
        print(f"Wilaya par défaut: {self.current_wilaya}")
        return self.current_wilaya

    def get_current_season(self):
        """Détermine la saison actuelle."""
        today = date.today()
        year = today.year

        seasons = {
            "Printemps": (date(year, 3, 21), date(year, 6, 20)),
            "Été": (date(year, 6, 21), date(year, 9, 22)),
            "Automne": (date(year, 9, 23), date(year, 12, 20))
        }
        
        for season_name, (start, end) in seasons.items():
            if start <= today <= end:
                self.current_season = season_name
                return self.current_season
        
        self.current_season = "Hiver"
        return self.current_season

    def _prepare_items_data(self, items_df):
        """Prépare les données des lieux pour la prédiction."""
        place_id_encoder = self.saved_data['place_id_encoder']
        items_df['place_id_enc'] = place_id_encoder.transform(items_df['_id'])
        return items_df
    
    def _get_encoded_user_id(self, user_id):
        """Récupère l'ID utilisateur encodé pour le modèle."""
        try:
            user_classes = self.saved_data['user_id_encoder'].classes_
            if str(user_id) in user_classes:
                return self.saved_data['user_id_encoder'].transform([str(user_id)])[0]
            return None
        except Exception as e:
            print(f"Erreur encodage utilisateur {user_id}: {e}")
            return None
    
    def _create_user_features(self, user_id):
        """Crée un vecteur de caractéristiques pour un nouvel utilisateur."""
        try:
            # Récupération des données utilisateur
            user_doc = self.db.users.find_one({"_id": user_id})
            if not user_doc:
                return None
                
            user_prefs = self.db.user_preferences.find_one({"user_id": user_id})
            
            # Création du vecteur de base
            feature_dim = self.saved_data['user_features'].shape[1]
            feature_vector = np.zeros(feature_dim)
            
            # Informations démographiques
            if 'age' in user_doc and user_doc['age']:
                normalized_age = (max(10, min(100, user_doc['age'])) - 10) / 90
                feature_vector[0] = normalized_age
                
            if 'sexe' in user_doc:
                feature_vector[1] = 1 if user_doc['sexe'].upper() == 'F' else 0
            
            # Préférences utilisateur
            if user_prefs and 'preferences' in user_prefs:
                prefs = user_prefs['preferences']
                top_prefs = sorted(
                    [(k, v) for k, v in prefs.items() if isinstance(v, (int, float))],
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                for i, (_, pref_value) in enumerate(top_prefs):
                    if i + 2 < len(feature_vector):
                        feature_vector[i + 2] = min(1.0, float(pref_value))
            
            # Application de la normalisation
            if 'user_scaler' in self.saved_data:
                scaler = self.saved_data['user_scaler']
                feature_vector = scaler.transform(
                    feature_vector.reshape(1, -1)
                ).flatten()
                
            return feature_vector
            
        except Exception as e:
            print(f"Erreur création caractéristiques utilisateur: {e}")
            return None
    
    def _find_similar_users(self, user_features, top_k=5):
        """Trouve les utilisateurs les plus similaires dans l'ensemble d'entraînement."""
        try:
            if self.saved_data['user_features'].shape[0] == 0:
                return np.array([])
                
            similarities = cosine_similarity(
                user_features.reshape(1, -1), 
                self.saved_data['user_features']
            )
            
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            return top_indices
            
        except Exception as e:
            print(f"Erreur recherche utilisateurs similaires: {e}")
            return np.array([])

    def _get_filtered_items(self, wilaya_name, season):
        """Récupère les lieux filtrés selon la wilaya et la saison."""
        query = {'wilaya': wilaya_name}
        
        if season:
            season_regex = season.lower()
            query['best_season'] = {'$regex': season_regex, '$options': 'i'}
        
        filtered_items = list(self.db.lieux.find(query))
        
        if not filtered_items:
            print(f"Aucun lieu trouvé pour {wilaya_name} en {season}")
            return None, None
        
        items_df = pd.DataFrame(filtered_items)
        items_df = self._prepare_items_data(items_df)
        wilaya_indices = torch.tensor(
            items_df['place_id_enc'].values, 
            dtype=torch.long
        ).to(self.device)
        
        return filtered_items, wilaya_indices

    def _predict_for_existing_user(self, user_id_enc, wilaya_indices):
        """Génère des prédictions pour un utilisateur existant."""
        user_tensor = torch.full(
            (len(wilaya_indices),), 
            user_id_enc, 
            dtype=torch.long
        ).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(
                self.graph.x, user_tensor, wilaya_indices
            ).cpu().numpy()
            
        return predictions

    def _predict_for_new_user(self, user_id, wilaya_indices):
        """Génère des prédictions pour un nouvel utilisateur."""
        user_features = self._create_user_features(user_id)
        if user_features is None:
            return None
        
        similar_users = self._find_similar_users(user_features, top_k=5)
        if len(similar_users) == 0:
            return None
        
        all_predictions = []
        for similar_user_enc in similar_users:
            user_tensor = torch.full(
                (len(wilaya_indices),), 
                similar_user_enc, 
                dtype=torch.long
            ).to(self.device)
            
            with torch.no_grad():
                preds = self.model(
                    self.graph.x, user_tensor, wilaya_indices
                ).cpu().numpy()
                all_predictions.append(preds)
        
        return np.mean(all_predictions, axis=0)

    def _format_recommendations(self, items, predictions, min_rating=DEFAULT_MIN_RATING, max_count=MAX_RECOMMENDATIONS):
        """Formate les recommandations avec filtrage et tri."""
        recommendations = []
        
        for item, pred in zip(items, predictions):
            if pred >= min_rating:
                rec = item.copy()
                rec['_id'] = str(rec['_id'])
                rec['place_id'] = str(rec['_id'])
                rec['predicted_rating'] = float(pred)
                recommendations.append(rec)
        
        # Tri par note décroissante et limitation
        recommendations.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return recommendations[:max_count]

    def generate_recommendations(self, wilaya_name=None, season=None, 
                               min_rating=DEFAULT_MIN_RATING):
        """
        Génère des recommandations pour tous les utilisateurs.
        
        Args:
            wilaya_name (str): Nom de la wilaya (détection auto si None)
            season (str): Saison (détection auto si None)
            min_rating (float): Note minimale pour les recommandations
            
        Returns:
            dict: Recommandations structurées pour tous les utilisateurs
        """
        # Configuration des paramètres
        self.current_wilaya = wilaya_name or self.detect_wilaya()
        self.current_season = season or self.get_current_season()
        
        print(f"\n{'='*60}")
        print(f"GÉNÉRATION DES RECOMMANDATIONS")
        print(f"{'='*60}")
        print(f"Wilaya: {self.current_wilaya}")
        print(f"Saison: {self.current_season}")
        print(f"Note minimale: {min_rating}")
        print(f"Maximum par utilisateur: {self.MAX_RECOMMENDATIONS}")
        
        # Récupération des lieux filtrés
        filtered_items, wilaya_indices = self._get_filtered_items(
            self.current_wilaya, self.current_season
        )
        
        if filtered_items is None:
            return None
        
        print(f"Lieux trouvés: {len(filtered_items)}")
        
        # Récupération des utilisateurs
        users = list(self.db.users.find(
            {}, {'_id': 1, 'prenom': 1, 'nom': 1, 'age': 1, 'sexe': 1}
        ))
        
        if not users:
            print("Aucun utilisateur trouvé!")
            return None
        
        print(f"Utilisateurs à traiter: {len(users)}")
        
        # Structure de retour
        results = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "wilaya": self.current_wilaya,
                "season": self.current_season,
                "min_rating": min_rating,
                "max_recommendations": self.MAX_RECOMMENDATIONS,
                "total_users": len(users),
                "total_items": len(filtered_items)
            },
            "recommendations": {}
        }
        
        # Traitement de chaque utilisateur
        successful_recs = 0
        
        for user in tqdm(users, desc="Génération des recommandations"):
            user_id = str(user["_id"])
            user_name = f"{user.get('prenom', '')} {user.get('nom', '')}".strip()
            
            # Tentative de prédiction
            user_id_enc = self._get_encoded_user_id(user_id)
            
            if user_id_enc is not None:
                # Utilisateur existant
                predictions = self._predict_for_existing_user(user_id_enc, wilaya_indices)
                user_type = "existing"
            else:
                # Nouvel utilisateur
                predictions = self._predict_for_new_user(user["_id"], wilaya_indices)
                user_type = "new"
            
            if predictions is None:
                continue
            
            # Formatage des recommandations
            user_recs = self._format_recommendations(
                filtered_items, predictions, min_rating, self.MAX_RECOMMENDATIONS
            )
            
            if user_recs:
                results["recommendations"][user_id] = {
                    "user_name": user_name,
                    "user_type": user_type,
                    "recommendations": user_recs,
                    "count": len(user_recs)
                }
                successful_recs += 1
        
        # Statistiques finales
        print(f"\n{'='*60}")
        print(f"RÉSULTATS")
        print(f"{'='*60}")
        print(f"Utilisateurs traités: {len(users)}")
        print(f"Recommandations générées: {successful_recs}")
        print(f"Taux de succès: {successful_recs/len(users)*100:.1f}%")
        
        results["metadata"]["successful_recommendations"] = successful_recs
        results["metadata"]["success_rate"] = successful_recs / len(users)
        
        return results

    def get_user_recommendations(self, user_id, wilaya_name=None, season=None, 
                                min_rating=DEFAULT_MIN_RATING):
        """
        Génère des recommandations pour un utilisateur spécifique.
        
        Args:
            user_id: ID de l'utilisateur
            wilaya_name (str): Nom de la wilaya
            season (str): Saison
            min_rating (float): Note minimale
            
        Returns:
            dict: Recommandations pour l'utilisateur spécifié
        """
        # Configuration
        self.current_wilaya = wilaya_name or self.detect_wilaya()
        self.current_season = season or self.get_current_season()
        
        # Récupération des lieux
        filtered_items, wilaya_indices = self._get_filtered_items(
            self.current_wilaya, self.current_season
        )
        
        if filtered_items is None:
            return None
        
        # Récupération des informations utilisateur
        user_doc = self.db.users.find_one({"_id": user_id})
        if not user_doc:
            print(f"Utilisateur {user_id} non trouvé")
            return None
        
        user_name = f"{user_doc.get('prenom', '')} {user_doc.get('nom', '')}".strip()
        
        # Prédiction
        user_id_enc = self._get_encoded_user_id(str(user_id))
        
        if user_id_enc is not None:
            predictions = self._predict_for_existing_user(user_id_enc, wilaya_indices)
            user_type = "existing"
        else:
            predictions = self._predict_for_new_user(user_id, wilaya_indices)
            user_type = "new"
        
        if predictions is None:
            return None
        
        # Formatage
        recommendations = self._format_recommendations(
            filtered_items, predictions, min_rating, self.MAX_RECOMMENDATIONS
        )
        
        return {
            "user_id": str(user_id),
            "user_name": user_name,
            "user_type": user_type,
            "wilaya": self.current_wilaya,
            "season": self.current_season,
            "recommendations": recommendations,
            "count": len(recommendations),
            "generation_date": datetime.now().isoformat()
        }

    def close(self):
        """Ferme la connexion à la base de données."""
        if self.client:
            self.client.close()
            print("Connexion MongoDB fermée.")
