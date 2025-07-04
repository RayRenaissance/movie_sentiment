from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Sample data
reviews = [
    "I loved the movie",
    "Absolutely brilliant storytelling",
    "Fantastic acting and plot",
    "An emotional rollercoaster",
    "This movie was amazing",
    "I really enjoyed it",
    "Great direction and screenplay",
    "Incredible performances by the cast",
    "The visuals were stunning",
    "A masterpiece of modern cinema",
    "One of the best films I've seen",
    "I hated the movie",
    "This movie was terrible",
    "Worst film ever",
    "Very boring and slow",
    "Bad acting and weak story",
    "Not worth the time",
    "Poorly directed and executed",
    "I fell asleep watching it",
    "The plot made no sense",
    "A total disappointment",
    "I would not recommend this movie"



]
labels = [1]*11 + [0]*11  # First 11 are positive (1), next 11 negative (0)

# Model pipeline
model = make_pipeline(CountVectorizer(), LogisticRegression())
model.fit(reviews, labels)

# Save the model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
