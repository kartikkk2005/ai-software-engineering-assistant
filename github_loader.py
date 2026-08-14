import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple
from config import REPOS_DIR

# Ignored directory names
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env", 
    ".idea", ".vscode", "dist", "build", "target", "vendor", ".cache"
}

# Ignored file extensions (binary, media, compiled assets)
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".tar", 
    ".gz", ".7z", ".rar", ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", 
    ".db", ".sqlite", ".bin", ".lock", ".icns", ".ttf", ".woff", ".woff2"
}

def parse_repo_url(repo_url_or_name: str) -> Tuple[str, str]:
    """Extracts owner and repository name from GitHub URL or short identifier."""
    clean_str = repo_url_or_name.strip().rstrip("/")
    if clean_str.endswith(".git"):
        clean_str = clean_str[:-4]
        
    if "github.com/" in clean_str:
        parts = clean_str.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    elif "/" in clean_str:
        parts = clean_str.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
            
    # Fallback to single string name
    sanitized = clean_str.replace(":", "_").replace("\\", "_").replace("/", "_")
    return "local", sanitized

def clone_or_download_repo(repo_url: str) -> Path:
    """Clones a GitHub repository or downloads its zip archive into REPOS_DIR."""
    owner, repo_name = parse_repo_url(repo_url)
    target_dir = REPOS_DIR / repo_name
    
    if target_dir.exists():
        print(f"[github_loader] Repository directory '{target_dir}' already exists. Using existing files.")
        return target_dir

    # Method 1: Try GitPython clone
    try:
        from git import Repo
        git_url = f"https://github.com/{owner}/{repo_name}.git" if owner != "local" else repo_url
        print(f"[github_loader] Cloning repository from {git_url}...")
        Repo.clone_from(git_url, target_dir, depth=1)
        print(f"[github_loader] Successfully cloned repository into '{target_dir}'.")
        return target_dir
    except Exception as e:
        print(f"[github_loader] Git clone failed ({e}). Attempting ZIP download fallback...")

    # Method 2: Fallback ZIP Download for GitHub
    if owner != "local":
        for branch in ["main", "master"]:
            zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"
            zip_path = REPOS_DIR / f"{repo_name}.zip"
            try:
                print(f"[github_loader] Downloading ZIP from {zip_url}...")
                urllib.request.urlretrieve(zip_url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(REPOS_DIR)
                
                extracted_folder = REPOS_DIR / f"{repo_name}-{branch}"
                if extracted_folder.exists():
                    extracted_folder.rename(target_dir)
                
                if zip_path.exists():
                    zip_path.unlink()
                    
                print(f"[github_loader] Successfully extracted repository into '{target_dir}'.")
                return target_dir
            except Exception as zip_err:
                print(f"[github_loader] ZIP download for branch '{branch}' failed: {zip_err}")
                if zip_path.exists():
                    zip_path.unlink()

    raise RuntimeError(f"Failed to load repository from '{repo_url}'. Please check URL or internet connection.")

def get_codebase_files(repo_path: Path) -> List[Path]:
    """Recursively collects all readable code & document file paths, skipping ignored patterns."""
    valid_files = []
    if not repo_path.exists():
        return valid_files
        
    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        
        for file_name in files:
            file_path = Path(root) / file_name
            if file_name.startswith("."):
                continue
            if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                continue
            valid_files.append(file_path)
            
    return sorted(valid_files)
