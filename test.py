from flask import Flask
from pymongo import MongoClient

app = Flask(__name__)

# Remplace par ton URI MongoDB Atlas
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)

# Remplace 'testdb' par le nom de ta base
db = client["Algeria"]

@app.route("/")
def home():
    try:
        # Essaye de lire une collection
        data = db["lieux"].find_one()  # test_collection peut être vide
        return f"Connexion MongoDB réussie ! Donnée : {data}"
    except Exception as e:
        return f"Erreur de connexion : {e}"

if __name__ == "__main__":
    app.run(debug=True)
