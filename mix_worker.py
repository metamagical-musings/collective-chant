#!/usr/bin/env python3
import os, sys, time, shutil, re, json

def main():
    if len(sys.argv) < 5:
        print("Usage: mix_worker.py <mix_id> <uploads_dir> <results_dir> <dummy_source_path>")
        sys.exit(1)
    mix_id = sys.argv[1]
    uploads_dir = sys.argv[2]
    results_dir = sys.argv[3]
    dummy_source = sys.argv[4]
    print(f"[Worker {mix_id}] Starting processing...")
    os.makedirs(results_dir, exist_ok=True)
    pattern = re.compile(r'^([^\.]+)\.([^\.]+)\.(.+)\.webm$')
    contributors = []
    files_processed = 0
    try:
        files = os.listdir(uploads_dir)
        if not files:
            print(f"[Worker {mix_id}] Uploads directory is empty.")
            # Decide: Exit or create empty mix? Exiting for now.
            sys.exit(0)
        for fname in files:
            if fname.endswith('.webm'):
                match = pattern.match(fname)
                if match:
                    token = match.group(2)
                    ip = match.group(3)
                    contributors.append({token: ip})
                    files_processed += 1
                else: print(f"[Worker {mix_id}] Warning: Filename does not match expected pattern: {fname}")
    except FileNotFoundError:
        print(f"[Worker {mix_id}] ERROR: Uploads directory not found: {uploads_dir}")
        sys.exit(1)
    if files_processed == 0:
        print(f"[Worker {mix_id}] No valid audio files found to process.")
        sys.exit(1)
    print(f"[Worker {mix_id}] Found {files_processed} contributors.")
    print(f"[Worker {mix_id}] Sample contributors: {contributors[:5]}")
    contributors_file_path = os.path.join(results_dir, f"{mix_id}.json")
    try:
        with open(contributors_file_path, 'w') as f: json.dump(contributors, f)
        print(f"[Worker {mix_id}] Contributor list saved to: {contributors_file_path}")
    except IOError as e:
        print(f"[Worker {mix_id}] ERROR writing contributor file: {e}")
        sys.exit(1)
    output_path = os.path.join(results_dir, f"{mix_id}.webm")
    if os.path.exists(dummy_source):
        shutil.copy2(dummy_source, output_path)
        print(f"[Worker {mix_id}] Dummy mix created at: {output_path}")
    else:
        print(f"[Worker {mix_id}] ERROR: Dummy source not found at {dummy_source}")
        sys.exit(1)
    for filename in files:
        if filename.endswith('.webm'):
            file_path = os.path.join(uploads_dir, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except FileNotFoundError:
                print(f"File not found (may have been deleted): {filename}")
            except PermissionError:
                print(f"Permission denied: {filename}")
    print(f"[Worker {mix_id}] Process complete.")
    sys.exit(0)
if __name__ == "__main__":
    main()
