import os
import sys
from huggingface_hub import create_repo, upload_folder, upload_file, whoami, HfApi
from termcolor import colored

import argparse

def upload_model_to_hf(
    repo_id: str,
    model_path: str,
    code_path: str = None,
    private: bool = False
):
    """
    Upload model checkpoint and code to Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/TARA" or just "TARA")
        model_path: Path to the model checkpoint directory
        code_path: Path to the code directory (defaults to current directory's public folder)
        private: Whether the repository should be private (default: False, i.e., public)
    """
    # Check if user is logged in and get username
    try:
        user_info = whoami()
        username = user_info.get('name', '')
        if not username:
            raise ValueError("Could not determine Hugging Face username. Please ensure you're logged in with `huggingface-cli login`")
        
        # Display user information
        print(colored("=" * 60, 'cyan'))
        print(colored("Hugging Face Authentication Check", 'cyan', attrs=['bold']))
        print(colored("=" * 60, 'cyan'))
        print(colored(f"✓ Logged in as: {username}", 'green'))
        if user_info.get('fullname'):
            print(colored(f"  Full name: {user_info.get('fullname')}", 'white'))
        print(colored(f"  User ID: {user_info.get('id', 'N/A')}", 'white'))
        print(colored("=" * 60, 'cyan'))
        
    except Exception as e:
        raise ValueError(f"Not logged in to Hugging Face. Please run `huggingface-cli login` first. Error: {e}")
    
    # If repo_id doesn't contain a '/', prepend username
    if '/' not in repo_id:
        repo_id = f"{username}/{repo_id}"
    
    print(colored(f"\n📦 Repository: {repo_id}", 'cyan', attrs=['bold']))
    print(colored(f"📁 Model path: {model_path}", 'white'))
    print(colored(f"📂 Code path: {code_path if code_path else 'public/ (default)'}", 'white'))
    print(colored(f"🔒 Private: {private}", 'white'))
    
    # Create repository if it doesn't exist
    print(colored(f"[1/4] Creating/checking repository: {repo_id}", 'cyan'))
    try:
        create_repo(repo_id, exist_ok=True, private=private)
        print(colored(f"✓ Repository {repo_id} is ready", 'green'))
    except Exception as e:
        print(colored(f"⚠ Error creating repo (might already exist): {e}", 'yellow'))
    
    # Upload model checkpoint files
    print(colored(f"[2/4] Uploading model checkpoint from {model_path}", 'cyan'))
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    try:
        # Folders to ignore during upload
        ignore_folders = ["embs", "metadata_results", "metrics", "__pycache__", ".git"]
        ignore_patterns = ["*.git*", "*.ipynb_checkpoints*"]
        
        # Add folder ignore patterns
        for folder in ignore_folders:
            ignore_patterns.append(f"**/{folder}/**")
            ignore_patterns.append(f"{folder}/**")
        
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
        )
        print(colored("✓ Model checkpoint uploaded successfully", 'green'))
    except Exception as e:
        print(colored(f"✗ Error uploading checkpoint: {e}", 'red'))
        raise
    
    # Upload code files
    if code_path is None:
        # Default to public folder in the same directory as this script
        code_path = os.path.dirname(os.path.abspath(__file__))
    
    print(colored(f"[3/4] Uploading code files from {code_path}", 'cyan'))
    
    # Files to upload from code directory
    code_files_to_upload = []
    for root, dirs, files in os.walk(code_path):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            file_path = os.path.join(root, file)
            # Skip certain files
            if file.endswith('.pyc') or file.startswith('.'):
                continue
            # Skip assets folder (videos/images)
            if 'assets' in file_path:
                continue
            # Skip the upload script itself
            if file == 'upload_on_huggingface.py':
                continue
            
            rel_path = os.path.relpath(file_path, code_path)
            code_files_to_upload.append((file_path, rel_path))
    
    # Upload each code file
    for local_path, path_in_repo in code_files_to_upload:
        try:
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print(colored(f"  ✓ Uploaded {path_in_repo}", 'green'))
        except Exception as e:
            print(colored(f"  ✗ Error uploading {path_in_repo}: {e}", 'red'))
    
    print(colored("✓ Code files uploaded successfully", 'green'))
    
    # Upload README if it exists
    print(colored("[4/4] Checking for README.md", 'cyan'))
    readme_path = os.path.join(code_path, "README.md")
    if os.path.exists(readme_path):
        try:
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print(colored("✓ Uploaded README.md", 'green'))
        except Exception as e:
            print(colored(f"⚠ Could not upload README: {e}", 'yellow'))
    else:
        print(colored("⚠ README.md not found in code directory", 'yellow'))
    
    print(colored(f"\n🎉 Upload complete! Model available at: https://huggingface.co/{repo_id}", 'green', attrs=['bold']))


def delete_files_from_repo(repo_id: str, paths_to_delete: list):
    """
    Delete files or folders from a Hugging Face repository (REMOTE ONLY - does NOT affect local files).
    
    ⚠️ IMPORTANT: This function ONLY deletes files from the Hugging Face repository.
    It does NOT delete anything from your local filesystem. Your local files are safe!
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/TARA")
        paths_to_delete: List of file/folder paths to delete from the REMOTE repository 
                        (e.g., ["embs/", "metadata_results/", "metrics/"])
    """
    # Check if user is logged in
    try:
        user_info = whoami()
        username = user_info.get('name', '')
        if not username:
            raise ValueError("Could not determine Hugging Face username. Please ensure you're logged in with `huggingface-cli login`")
    except Exception as e:
        raise ValueError(f"Not logged in to Hugging Face. Please run `huggingface-cli login` first. Error: {e}")
    
    # If repo_id doesn't contain a '/', prepend username
    if '/' not in repo_id:
        repo_id = f"{username}/{repo_id}"
    
    print(colored("=" * 60, 'yellow'))
    print(colored("⚠️  REMOTE DELETION ONLY - Local files are SAFE!", 'yellow', attrs=['bold']))
    print(colored("=" * 60, 'yellow'))
    print(colored(f"Deleting files/folders from REMOTE repository: {repo_id}", 'cyan'))
    print(colored("(Your local files will NOT be affected)", 'green'))
    api = HfApi()
    
    deleted_count = 0
    for path in paths_to_delete:
        try:
            # Ensure path ends with / for folders or remove trailing / for files
            path_clean = path.rstrip('/')
            print(colored(f"  Deleting: {path_clean}...", 'yellow'))
            
            # Delete the file/folder
            api.delete_file(
                path_in_repo=path_clean,
                repo_id=repo_id,
                repo_type="model",
            )
            print(colored(f"  ✓ Deleted: {path_clean}", 'green'))
            deleted_count += 1
        except Exception as e:
            print(colored(f"  ✗ Error deleting {path_clean}: {e}", 'red'))
            print(colored("    (File/folder may not exist or may have already been deleted)", 'yellow'))
    
    if deleted_count > 0:
        print(colored(f"\n✓ Successfully deleted {deleted_count} item(s)!", 'green'))
        print(colored(f"  Repository: https://huggingface.co/{repo_id}", 'cyan'))
    else:
        print(colored("\n⚠ No files were deleted.", 'yellow'))


def add_collaborator_to_repo(repo_id: str, collaborator_username: str):
    """
    Add a collaborator to a private Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., "username/TARA")
        collaborator_username: Hugging Face username of the collaborator to add
    """
    # Check if user is logged in
    try:
        user_info = whoami()
        username = user_info.get('name', '')
        if not username:
            raise ValueError("Could not determine Hugging Face username. Please ensure you're logged in with `huggingface-cli login`")
    except Exception as e:
        raise ValueError(f"Not logged in to Hugging Face. Please run `huggingface-cli login` first. Error: {e}")
    
    # If repo_id doesn't contain a '/', prepend username
    if '/' not in repo_id:
        repo_id = f"{username}/{repo_id}"
    
    print(colored(f"Adding collaborator '{collaborator_username}' to {repo_id}...", 'cyan'))
    try:
        api = HfApi()
        api.add_colaborator(repo_id=repo_id, username=collaborator_username, repo_type="model")
        print(colored(f"✓ Successfully added {collaborator_username} as a collaborator!", 'green'))
        print(colored(f"  They can now access the repository at: https://huggingface.co/{repo_id}", 'green'))
    except Exception as e:
        print(colored(f"✗ Error adding collaborator: {e}", 'red'))
        print(colored("  Note: You can also add collaborators via the web interface:", 'yellow'))
        print(colored(f"  https://huggingface.co/{repo_id}/settings", 'yellow'))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload TARA model to Hugging Face Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        default='/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint',
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--code_path",
        type=str,
        default=None,
        help="Path to the code directory (defaults to public folder containing this script)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="TARA",
        help="Hugging Face repository ID (e.g., 'username/TARA' or just 'TARA')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--add_collaborator",
        type=str,
        default=None,
        help="Add a collaborator to the repository (provide their Hugging Face username)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed with upload"
    )
    parser.add_argument(
        "--delete",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Delete files/folders from the REMOTE Hugging Face repository only (NOT local files). Example: --delete embs/ metadata_results/ metrics/"
    )
    
    args = parser.parse_args()
    
    # Verify username before proceeding
    try:
        user_info = whoami()
        detected_username = user_info.get('name', '')
        if not detected_username:
            raise ValueError("Could not determine Hugging Face username.")
        
        print(colored("\n" + "=" * 60, 'cyan'))
        print(colored("Pre-Upload Verification", 'cyan', attrs=['bold']))
        print(colored("=" * 60, 'cyan'))
        print(colored(f"Detected username: {detected_username}", 'green'))
        
        # Check if repo_id matches expected username
        final_repo_id = args.repo_id if '/' in args.repo_id else f"{detected_username}/{args.repo_id}"
        if '/' in args.repo_id:
            repo_username = args.repo_id.split('/')[0]
            if repo_username != detected_username:
                print(colored(f"⚠ WARNING: Repository username '{repo_username}' doesn't match detected username '{detected_username}'", 'yellow'))
                print(colored(f"   The repository will be created under: {final_repo_id}", 'yellow'))
        else:
            print(colored(f"Repository will be created as: {final_repo_id}", 'cyan'))
        
        print(colored("=" * 60, 'cyan'))
        
        # Ask for confirmation unless --yes flag is set
        if not args.yes:
            response = input(colored("\nProceed with upload? (yes/no): ", 'yellow'))
            if response.lower() not in ['yes', 'y']:
                print(colored("Upload cancelled.", 'red'))
                sys.exit(0)
        
    except Exception as e:
        raise ValueError(f"Could not verify Hugging Face authentication. Please run `huggingface-cli login` first. Error: {e}")
    
    upload_model_to_hf(
        repo_id=args.repo_id,
        model_path=args.model_path,
        code_path=args.code_path,
        private=args.private
    )
    
    # Delete files if specified
    if args.delete:
        delete_files_from_repo(
            repo_id=args.repo_id,
            paths_to_delete=args.delete
        )
        sys.exit(0)  # Exit after deletion
    
    # Add collaborator if specified
    if args.add_collaborator:
        add_collaborator_to_repo(
            repo_id=args.repo_id,
            collaborator_username=args.add_collaborator
        )
    