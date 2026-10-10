import hashlib

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

print("series_manifest.json:", hash_file("series_manifest.json"))
print("omega_values.json:", hash_file("omega_values.json"))