# train_resume_model.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Resume_train_dataset.csv")  # Ensure it has 'Resume' and 'Category' columns

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(df['Resume'], df['Category'], test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, 'resume_classifier.pkl')
print("\n🎉 Model trained and saved as 'resume_classifier.pkl'")
