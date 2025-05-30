import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchdiffeq import odeint_adjoint as odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from torch.utils.data import DataLoader, TensorDataset
from pymongo import MongoClient
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from tqdm import tqdm
import random
import json
from bson import ObjectId
import pickle
import unicodedata

warnings.filterwarnings('ignore')

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def normalize_string(s):
    if pd.isna(s):
        return ""
    return unicodedata.normalize('NFKD', str(s).lower()).encode('ascii', 'ignore').decode('utf-8').strip()

def load_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Algeria']
    
    users = list(db.users.find({}, {'_id': 1, 'prenom': 1, 'nom': 1, 'age': 1, 'sexe': 1, 'visited_places': 1}))
    users_df = pd.DataFrame(users)
    
    # Récupérer tous les champs des lieux
    items = list(db.lieux.find({}))
    items_df = pd.DataFrame(items)

    items_df['wilaya_normalized'] = items_df['wilaya'].apply(normalize_string)
    
    ratings = list(db.interactions.find({}, {'user_id': 1, 'place_id': 1, 'rating': 1}))
    ratings_df = pd.DataFrame(ratings)
    
    return users_df, items_df, ratings_df

def preprocess_data(users, items, ratings):
    ratings = ratings.dropna(subset=['rating'])
    ratings['rating'] = ratings['rating'].clip(1, 5)

    # Filter ratings to only those with place_id in items['_id']
    ratings = ratings[ratings['place_id'].isin(items['_id'])].reset_index(drop=True)
    
    # Fit encoders on the union of all IDs to avoid unseen label errors
    all_user_ids = pd.Series(list(set(ratings['user_id']).union(set(users['_id']))))
    user_id_encoder = LabelEncoder()
    user_id_encoder.fit(all_user_ids)
    ratings['user_id_enc'] = user_id_encoder.transform(ratings['user_id'])
    users['user_id_enc'] = user_id_encoder.transform(users['_id'])
    
    all_place_ids = pd.Series(list(set(ratings['place_id']).union(set(items['_id']))))
    place_id_encoder = LabelEncoder()
    place_id_encoder.fit(all_place_ids)
    ratings['place_id_enc'] = place_id_encoder.transform(ratings['place_id'])
    items['place_id_enc'] = place_id_encoder.transform(items['_id'])

    # Ensure all place_id_enc are within the correct range
    max_place_id_enc = items['place_id_enc'].max()
    assert ratings['place_id_enc'].max() <= max_place_id_enc, \
        f"Found place_id_enc in ratings out of bounds: max={ratings['place_id_enc'].max()}, allowed={max_place_id_enc}"

    # User features
    users['age'] = users['age'].fillna(users['age'].median()).clip(10, 100)
    users['sexe_enc'] = users['sexe'].apply(lambda x: 1 if x == 'F' else 0)
    
    all_crowd_types = [
        "Familial", "Lieux pour les groupes d'amis", "Lieux pour les Jeunes Filles",
        "Lieux pour les Couples", "Lieux pour les âgés", "Lieux pour les Sportifs",
        "Lieux pour Personnes en situation de handicap"
    ]
    
    crowd_mapping = {t: t.split()[-1].replace("'", "")[:5] for t in all_crowd_types}
    
    user_prefs = defaultdict(lambda: defaultdict(float))
    for _, row in ratings.iterrows():
        place_id = row['place_id']
        place_info_df = items[items['_id'] == place_id]
        if place_info_df.empty:
            continue  # Skip if no matching place
        place_info = place_info_df.iloc[0]
        crowd_types = place_info['characteristics']
        note = row['rating']
        
        for crowd_type in all_crowd_types:
            if crowd_type in crowd_types:
                user_prefs[row['user_id']][crowd_mapping[crowd_type]] += (note - 2.5)
    
    for cat in crowd_mapping.values():
        users[cat] = 0.0
    
    for user_id, prefs in user_prefs.items():
        total = max(1, sum(abs(v) for v in prefs.values()))
        for cat, val in prefs.items():
            users.loc[users['_id'] == user_id, cat] = val / total
    
    user_features = users[['age', 'sexe_enc'] + list(crowd_mapping.values())].fillna(0)
    user_scaler = MinMaxScaler()
    user_features = user_scaler.fit_transform(user_features)
    
    # Item features
    items['categorie'] = items['category']
    items['sous_categorie'] = items['subcategory']
    items['cat_combine'] = items['categorie'] + "_" + items['sous_categorie']
    cat_encoder = LabelEncoder()
    items['cat_enc'] = cat_encoder.fit_transform(items['cat_combine'])
    
    items['visit_time'] = items['average_visit_time'].fillna(2).clip(0.5, 8)
    
    def parse_entry_fee(fee):
        if isinstance(fee, (int, float)):
            return float(fee)
        elif isinstance(fee, str):
            # Handle cases like "150da"
            fee = fee.lower().replace('da', '').strip()
            try:
                return float(fee)
            except ValueError:
                return 0
        return 0

    items['frais_entree'] = items['entry_fee'].apply(parse_entry_fee).clip(0, 5000)
    
    place_popularity = ratings['place_id'].value_counts().to_dict()
    items['popularity'] = items['_id'].map(lambda x: place_popularity.get(x, 0))
    
    item_features = items[['cat_enc', 'visit_time', 'frais_entree', 'popularity']]
    item_scaler = MinMaxScaler()
    item_features = item_scaler.fit_transform(item_features)
    
    return (users, items, ratings, 
            user_features.astype(np.float32), 
            item_features.astype(np.float32), 
            user_id_encoder, place_id_encoder,
            user_scaler, item_scaler, crowd_mapping, cat_encoder)

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
            # Disable BatchNorm during inference
            self.encoder[1].eval()  # Set BatchNorm to evaluation mode
        return self.encoder(x), None

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
            # Disable BatchNorm during inference
            self.encoder[1].eval()  # Set BatchNorm to evaluation mode
        return self.encoder(x), None

def build_graph(ratings, user_features, item_features, user_mlp, item_mlp):
    with torch.no_grad():
        user_tensor = torch.tensor(user_features, dtype=torch.float)
        user_embeddings, _ = user_mlp(user_tensor)
        
        item_tensor = torch.tensor(item_features, dtype=torch.float)
        item_embeddings, _ = item_mlp(item_tensor)
    
    x = torch.cat([
        F.normalize(user_embeddings, p=2, dim=1),
        F.normalize(item_embeddings, p=2, dim=1)
    ], dim=0)
    
    num_users = user_embeddings.shape[0]
    
    edge_index = torch.tensor([
        ratings['user_id_enc'].values,
        ratings['place_id_enc'].values + num_users
    ], dtype=torch.long)
    
    edge_attr = torch.tensor((ratings['rating'].values - 1) / 4, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), num_users

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

        # Calcul des messages concaténés + attention
        concat = torch.cat([src_emb, dst_emb, edge_attr], dim=1)
        attention_weights = torch.sigmoid(self.attention(concat))

        # Messages pondérés par attention
        messages = attention_weights * local_diff[src]

        # Agrégation des messages (somme)
        out = torch.zeros_like(x)
        out = out.index_add(0, dst, messages)

        # Normalisation par degré
        out = out / self.degree.unsqueeze(1)

        return out

class EnhancedGODEModel(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, graph):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        
        self.odeblock = ODEBlock(
            EnhancedODEFunc(latent_dim, graph.edge_index, graph.edge_attr),
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
        
        if torch.any(item_idx >= items.shape[0]):
            raise IndexError(f"item_idx contains out-of-bounds indices: max={item_idx.max().item()}, items.shape[0]={items.shape[0]}")
        
        user_emb = users[user_idx]
        item_emb = items[item_idx]
        
        combined = torch.cat([user_emb, item_emb], dim=1)
        pred = self.predictor(combined)
        return (pred * 4 + 1).squeeze()
    
def enhanced_evaluate(model, graph, test_data, num_users, user_id_encoder, place_id_encoder, items_df):
    model.eval()
    with torch.no_grad():
        test_users = torch.tensor(test_data['user_id_enc'].values, dtype=torch.long).to(device)
        test_items = torch.tensor(test_data['place_id_enc'].values, dtype=torch.long).to(device)
        test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float).to(device)
        
        preds = model(graph.x, test_users, test_items)
        
        # Créer un DataFrame avec les résultats
        results = pd.DataFrame({
            'user_id': user_id_encoder.inverse_transform(test_users.cpu()),
            'place_id': place_id_encoder.inverse_transform(test_items.cpu()),
            'real_rating': test_ratings.cpu().numpy(),
            'predicted_rating': preds.cpu().numpy()
        })
        
        # Ajouter les informations des lieux
        results = results.merge(items_df[['_id', 'name', 'wilaya']], 
                               left_on='place_id', right_on='_id', how='left')
        
        # Calcul des métriques
        mae = mean_absolute_error(results['real_rating'], results['predicted_rating'])
        rmse = sqrt(mean_squared_error(results['real_rating'], results['predicted_rating']))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'predictions': results[['user_id', 'place_id', 'name', 'wilaya', 'real_rating', 'predicted_rating']]
        }

def print_metrics(metrics, prefix, show_predictions=True):
    print(f"\n{prefix} Metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    if show_predictions and 'predictions' in metrics:
        print("\nSample predictions vs real ratings:")
        print(metrics['predictions'].head(10))

def enhanced_train(model, optimizer, scheduler, graph, train_data, val_data, num_users, user_id_encoder, place_id_encoder, items_df, epochs=100, batch_size=1024):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_dataset = TensorDataset(
        torch.tensor(train_data['user_id_enc'].values, dtype=torch.long),
        torch.tensor(train_data['place_id_enc'].values, dtype=torch.long),
        torch.tensor(train_data['rating'].values, dtype=torch.float)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for user_batch, item_batch, rating_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            pred = model(graph.x.to(device), 
                        user_batch.to(device), 
                        item_batch.to(device))
            
            loss = F.mse_loss(pred, rating_batch.to(device))
            loss += 0.001 * sum(p.pow(2.0).sum() for p in model.parameters())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation avec affichage des prédictions seulement à la dernière epoch
        val_results = enhanced_evaluate(model, graph, val_data, num_users, user_id_encoder, place_id_encoder, items_df)
        val_loss = val_results['mae']
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}")
        print_metrics(val_results, "Validation", show_predictions=(epoch == epochs-1))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_loss,
                'epoch': epoch
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
def save_recommendations_to_json(recommendations, filename='recommendations.json'):
    """Sauvegarde les recommandations dans un fichier JSON avec conversion des types NumPy"""
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, pd.Timestamp):
            return str(o)
        return o

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2, default=convert)
    print(f"\nRecommendations saved to {filename}")

def save_complete_model(model, user_mlp, item_mlp, graph, user_id_encoder, place_id_encoder,
                       user_scaler, item_scaler, crowd_mapping, cat_encoder,
                       user_features, item_features, val_results, test_results):
    """Save the complete model state with all necessary components"""
    save_dict = {
        # Model architecture
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_users': model.num_users,
            'num_items': model.num_items,
            'latent_dim': model.latent_dim,
            'user_feature_dim': user_features.shape[1],
            'item_feature_dim': item_features.shape[1]
        },
        
        # Encoders and scalers
        'user_id_encoder': user_id_encoder,
        'place_id_encoder': place_id_encoder,
        'user_scaler': user_scaler,
        'item_scaler': item_scaler,
        
        # Feature engineering
        'crowd_mapping': crowd_mapping,
        'cat_encoder': cat_encoder,
        
        # Graph data
        'graph_data': {
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr
        },
        
        # Feature matrices
        'user_features': user_features,
        'item_features': item_features,
        
        # MLP components
        'user_mlp_state_dict': user_mlp.state_dict(),
        'item_mlp_state_dict': item_mlp.state_dict(),
        
        # Evaluation
        'val_metrics': val_results,
        'test_metrics': test_results
    }
    
    # Save with pickle to handle scikit-learn objects
    with open('tourism_recommender_complete2.pkl', 'wb') as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nComplete model saved to tourism_recommender_complete.pkl")

def get_all_users_wilaya_recommendations(model, graph, users_df, items_df, user_id_encoder, 
                                       place_id_encoder, num_users, wilaya_name, min_rating=3.0):
    """Génère TOUTES les recommandations ≥3 pour chaque utilisateur"""
    wilaya_normalized = normalize_string(wilaya_name)
    wilaya_items = items_df[items_df['wilaya_normalized'] == wilaya_normalized]
    
    if wilaya_items.empty:
        print(f"\nAucun lieu trouvé pour la wilaya '{wilaya_name}'")
        return {}

    wilaya_indices = torch.tensor(wilaya_items['place_id_enc'].values, dtype=torch.long).to(device)
    all_recommendations = {}
    
    for user_id in users_df['_id']:
        try:
            user_id_enc = user_id_encoder.transform([user_id])[0]
            user_tensor = torch.full((len(wilaya_items),), user_id_enc, dtype=torch.long).to(device)
            
            with torch.no_grad():
                preds = model(graph.x, user_tensor, wilaya_indices).cpu().numpy()
            
            # Filtrer strictement par note >=3 sans limite de quantité
            recommendations = [
                {
                    'place_id': str(place_id_encoder.inverse_transform([wilaya_items.iloc[idx]['place_id_enc']])[0]),
                    'name': str(wilaya_items.iloc[idx]['name']),
                    'wilaya': str(wilaya_items.iloc[idx]['wilaya']),
                    'category': str(wilaya_items.iloc[idx]['category']),
                    'predicted_rating': float(pred)
                }
                for idx, pred in enumerate(preds) 
                if pred > min_rating
            ]

            
            # Tri par note décroissante
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            if recommendations:
                all_recommendations[str(user_id)] = recommendations
                
        except Exception as e:
            print(f"Erreur pour l'utilisateur {user_id}: {str(e)}")
    
    return all_recommendations

# Modifiez la fonction get_wilaya_recommendations pour filtrer par note > 3
def get_wilaya_recommendations(model, graph, user_id, items_df, user_id_encoder, 
                              place_id_encoder, num_users, wilaya_name, min_rating=3.0, top_k=30):
    model.eval()
    with torch.no_grad():
        wilaya_normalized = normalize_string(wilaya_name)
        wilaya_items = items_df[items_df['wilaya_normalized'] == wilaya_normalized]
        
        if wilaya_items.empty:
            print(f"\nDebug: No items found for wilaya '{wilaya_name}' (normalized: '{wilaya_normalized}')")
            print("Available wilayas:", items_df['wilaya'].unique())
            return []
        
        print(f"\nFound {len(wilaya_items)} places in {wilaya_name}")
        
        wilaya_indices = torch.tensor(wilaya_items['place_id_enc'].values, dtype=torch.long).to(device)
        user_tensor = torch.full((len(wilaya_items),), user_id, dtype=torch.long).to(device)
        preds = model(graph.x, user_tensor, wilaya_indices)
        
        recommendations = []
        for idx, pred in zip(wilaya_items.index, preds.cpu().numpy()):
            if pred > min_rating:
                original_place_id = place_id_encoder.inverse_transform([wilaya_items.loc[idx, 'place_id_enc']])[0]
                recommendations.append({
                    'place_data': {
                        '_id': str(original_place_id),  # Convertir en string
                        'name': str(wilaya_items.loc[idx, 'name']),
                        'wilaya': str(wilaya_items.loc[idx, 'wilaya']),
                        'category': str(wilaya_items.loc[idx, 'category'])
                    },
                    'predicted_rating': float(pred)  # Conversion explicite
                })
        
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:top_k]

def main():
    print("Loading and preprocessing data...")
    users, items, ratings = load_data()
    
    data = preprocess_data(users, items, ratings)
    users, items, ratings, user_features, item_features, user_id_encoder, place_id_encoder, user_scaler, item_scaler, crowd_mapping, cat_encoder = data
    
    latent_dim = 64
    user_mlp = EnhancedMLPEncoder(user_features.shape[1], latent_dim).to(device)
    item_mlp = EnhancedMLPEncoder(item_features.shape[1], latent_dim).to(device)
    
    print("\nBuilding graph...")
    graph, num_users = build_graph(ratings, user_features, item_features, user_mlp, item_mlp)
    graph = graph.to(device)
    
    # Split data
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42, stratify=ratings['rating'])
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=test_data['rating'])
    
    print("\nInitializing model...")
    model = EnhancedGODEModel(num_users, len(items), latent_dim, graph).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    print("\nStarting training...")
    enhanced_train(model, optimizer, scheduler, graph, train_data, val_data, num_users, user_id_encoder, place_id_encoder, items, epochs=100)
    
    # Evaluation finale
    print("\n===============================")
    print("Validation Results:")
    val_results = enhanced_evaluate(model, graph, val_data, num_users, user_id_encoder, place_id_encoder, items)
    print_metrics(val_results, "Validation")
    
    print("\n===============================")
    print("Test Results:")
    test_results = enhanced_evaluate(model, graph, test_data, num_users, user_id_encoder, place_id_encoder, items)
    print_metrics(test_results, "Test")
    
    print("\nGénération des recommandations pour Alger (note ≥3 uniquement)")
    alger_recommendations = get_all_users_wilaya_recommendations(
        model, graph, users, items, user_id_encoder, 
        place_id_encoder, num_users, wilaya_name='Alger', 
        min_rating=3.0  # Suppression du paramètre top_k
    )
    
    # Sauvegarde des recommandations
    save_recommendations_to_json(alger_recommendations, 'alger_recommendations.json')
    
    # Affichage des recommandations pour chaque utilisateur
    print("\nRecommendations for Alger:")
    for user_id, recs in alger_recommendations.items():
        if recs:  # Afficher seulement les utilisateurs avec des recommandations
            user_info = users[users['_id'] == user_id].iloc[0]
            print(f"\nUser: {user_info['prenom']} {user_info['nom']} (Age: {user_info['age']}, Sexe: {user_info['sexe']})")
            print("Top Recommendations:")
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec['name']} (Catégorie: {rec['category']}, Note prédite: {rec['predicted_rating']:.2f})")
    
    # Sauvegarde du modèle complet
    save_complete_model(
        model, user_mlp, item_mlp, graph, user_id_encoder, place_id_encoder,
        user_scaler, item_scaler, crowd_mapping, cat_encoder,
        user_features, item_features, val_results, test_results
    )
    
    # Test recommendations pour un utilisateur spécifique
    test_wilaya = 'Alger'
    test_user = users.iloc[0]['_id']
    user_id_enc = user_id_encoder.transform([test_user])[0]
    
    print(f"\nTesting recommendations for user {test_user} in {test_wilaya} (rating > 3)")
    recs = get_wilaya_recommendations(
        model, graph, user_id_enc, items,
        user_id_encoder, place_id_encoder, num_users,
        test_wilaya, min_rating=3.0
    )
    
    if recs:
        print("\nTop Recommendations:")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['place_data']['name']} (Predicted Rating: {rec['predicted_rating']:.2f})")
        save_recommendations_to_json(recs, f'recommendations_user_{test_user}.json')
    else:
        print("\nNo recommendations with rating > 3 generated. Check debug messages above.")

if __name__ == "__main__":
    main()