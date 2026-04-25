import importlib.metadata

packages = []

# Iterate through all installed packages
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    version = dist.version
    size_bytes = 0
    
    # dist.files contains all files installed by the package
    if dist.files:
        for file in dist.files:
            # Locate the absolute file path on your system
            file_path = dist.locate_file(file)
            try:
                if file_path.exists() and file_path.is_file():
                    size_bytes += file_path.stat().st_size
            except Exception:
                pass
                
    size_mb = size_bytes / (1024 * 1024)
    packages.append((name, version, size_mb))

# Sort packages by size in descending order (biggest first)
packages.sort(key=lambda x: x[2], reverse=True)

# Print the formatted output
print(f"{'Package':<30} | {'Version':<15} | {'Size (MB)'}")
print("-" * 60)
for name, version, size in packages:
    print(f"{name:<30} | {version:<15} | {size:.2f} MB")