
    #!/usr/bin/env python3
"""
Repository Analysis Tool

This script analyzes GitHub or GitLab repositories based on a specified role
and provides answers to predefined questions in a PDF document.

Usage:
    python main.py --url <repository_url> --role <role> [--persistent]
    python main.py --local-path <repository_path> --role <role>

Example:
    python main.py --url https://github.com/fastapi-users/fastapi-users --role programmer --persistent
"""
import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlparse

from repo_fetcher import RepoFetcher
from git import Repo
from parser_python_ts import extract_nodes_and_edges_from_python
from file_walker import find_files_by_extension


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze GitHub/GitLab repositories")
    parser.add_argument("--url")
    parser.add_argument("--local-path")

    parser.add_argument("--github-token")
    parser.add_argument("--gitlab-token")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-process", action="store_true")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", default="report.pdf")

    args = parser.parse_args()

    if not args.url and not args.local_path and not (args.skip_fetch and args.skip_process):
        parser.error("Either --url or --local-path is required")
    return args

def get_repo_key(url=None, local_path=None):
    if url and url.startswith("http"):
        parsed = urlparse(url)
        path = parsed.path.strip("/")  # e.g., andrivet/python-asn1
        return path.replace("/", "_").replace(".git", "")
    elif local_path:
        # Check if .git/config exists and parse remote origin
        try:
            repo = Repo(local_path)
            remote_url = next(repo.remote().urls)
            return get_repo_key(url=remote_url)
        except Exception:
            # fallback to folder name
            return os.path.basename(os.path.normpath(local_path))
    else:
        raise ValueError("Must provide --url or --local-path")

def setup_logging(log_level):
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"repo_analysis_{int(time.time())}.log")
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset.timer").setLevel(logging.WARNING)
    logging.getLogger("fontTools.ttLib.ttFont").setLevel(logging.WARNING)
    return logging.getLogger("main")

def setup_directories(base_dir, repo_name):
    repo_dir = os.path.join(base_dir, repo_name)
    data_dir = os.path.join(repo_dir, "data")
    embeddings_dir = os.path.join(repo_dir, "embeddings")
    reports_dir = os.path.join(repo_dir, "reports")
    for d in [repo_dir, data_dir, embeddings_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
    return {"repo": repo_dir, "data": data_dir, "embeddings": embeddings_dir, "reports": reports_dir}

def fetch_repository(args, logger):
    logger.info("Fetching repository...")
    with RepoFetcher(github_token=args.github_token, gitlab_token=args.gitlab_token) as fetcher:
        repo_path = fetcher.fetch_repo(url=args.url or "local", local_path=args.local_path, persistent=args.persistent)
        repo_info = fetcher.get_basic_repo_info()
        return repo_path, repo_info

def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    logger.info("Starting repository analysis")
    try:
        repo_path, repo_info = None, None



        if not args.skip_fetch:
            # Fetch the repository
            repo_path, repo_info = fetch_repository(args, logger)

            # Use the name from repo_info to derive a consistent repo_key
            repo_key = get_repo_key(url=args.url, local_path=args.local_path)

            # Setup directory structure
            dirs = setup_directories(args.output_dir, repo_key)

            # Save repo_info.json into data dir
            repo_info_path = os.path.join(dirs["data"], "repo_info.json")
            with open(repo_info_path, "w") as f:
                json.dump(repo_info, f, indent=2)

        else:
            # Skip fetch: infer repo_key from local path or URL
            repo_key = get_repo_key(url=args.url, local_path=args.local_path)
            dirs = setup_directories(args.output_dir, repo_key)

            # Load repo_info.json from data dir
            repo_info_path = os.path.join(dirs["data"], "repo_info.json")
            if not os.path.exists(repo_info_path):
                logger.error("Missing repo_info.json and --skip-fetch was used")
                sys.exit(1)

            with open(repo_info_path, "r") as f:
                repo_info = json.load(f)
            repo_path = args.local_path
        # ----------------------------
        # Test: Extract Nodes and Edges using Tree-sitter
        # ----------------------------
        if not args.skip_process:
            logger.info("Processing Python files with Tree-sitter")

            py_files = find_files_by_extension(repo_path, ['.py'])
            all_nodes, all_edges = [], []

            for file in py_files:
                nodes, edges = extract_nodes_and_edges_from_python(file)
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            # Log summary
            logger.info(f"Parsed {len(py_files)} Python files")
            logger.info(f"Extracted {len(all_nodes)} nodes and {len(all_edges)} edges")

            # Save as JSON for inspection (optional)
            with open(os.path.join(dirs["data"], "graph_nodes.json"), "w", encoding="utf-8") as f:
                json.dump([n.__dict__ for n in all_nodes], f, indent=2)

            with open(os.path.join(dirs["data"], "graph_edges.json"), "w", encoding="utf-8") as f:
                json.dump([e.__dict__ for e in all_edges], f, indent=2)

        # Process the repo content into chunks
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
