import pkg_resources
import os

def get_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

site_packages = pkg_resources.get_distribution('pip').location
packages = {dist.project_name: dist.location for dist in pkg_resources.working_set}

sizes = []
for package, path in packages.items():
    size_bytes = get_size(os.path.join(path, package))
    size_mb = round(size_bytes / (1024 * 1024), 2)
    sizes.append((package, size_mb))

# Trier par taille décroissante
sizes.sort(key=lambda x: x[1], reverse=True)

# Afficher les plus gros packages
for name, size in sizes:
    print(f"{name:<30}: {size} MB")
