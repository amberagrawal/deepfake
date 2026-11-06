import joblib
import pandas as pd
from pathlib import Path

model_path = Path("models/spectral_svm.joblib")
feature_path = Path("outputs/features/uadfv_features.csv")

print("========== TRAINED MODEL ==========")

model = joblib.load(model_path)

print("Model type:", type(model))
print("Model steps:", getattr(model, "named_steps", "No pipeline steps"))
print("Expected feature count:", getattr(model, "n_features_in_", "Not directly available"))

if hasattr(model, "feature_names_in_"):
    print("\nModel feature names:")
    print(list(model.feature_names_in_))

print("\n========== UADFV FEATURE CSV ==========")

df = pd.read_csv(feature_path)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(i, col)

print("\n========== TRAINING SCRIPT ==========")

script = Path("src/train_classifier.py")
if script.exists():
    print(script.read_text(encoding="utf-8"))
else:
    print("Training script not found.")
