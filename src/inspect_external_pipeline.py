import joblib
from pathlib import Path

model_data = joblib.load("models/spectral_svm.joblib")

print("========== SAVED MODEL CONTENTS ==========")
print("Dictionary keys:", list(model_data.keys()))

model = model_data["model"]
feature_columns = model_data["feature_columns"]

print("\nModel pipeline:", model)
print("\nFeature count:", len(feature_columns))
print("First 10 features:", feature_columns[:10])
print("Last 10 features:", feature_columns[-10:])
print("Class mapping:", model_data["class_mapping"])

print("\n========== FEATURE EXTRACTOR ==========")
extractor = Path("src/batch_feature_extractor.py")

if extractor.exists():
    print(extractor.read_text(encoding="utf-8"))
else:
    print("batch_feature_extractor.py not found")

print("\n========== VIDEO PROCESSING SETTINGS ==========")
for filename in [
    "src/batch_feature_extractor.py",
    "src/feature_extractor.py",
    "src/graph_features.py"
]:
    path = Path(filename)
    if path.exists():
        print(f"\nFound: {filename}")
