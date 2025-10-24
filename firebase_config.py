import firebase_admin
from firebase_admin import credentials, firestore

# Path to your downloaded Firebase service account key
cred = credentials.Certificate("serviceAccountKey.json")

# Initialize Firebase app only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()
