#!/usr/bin/env python3
"""
dedupe_copy.py - Production-grade photo deduplication and copy tool.

Compares two large photo directories and copies only files from Target
that are NOT present in Reference, using a two-pass duplicate detection strategy:
  - PASS A: Exact byte-identical detection (blake2b hash)
  - PASS B: Perceptual hash detection for images (pHash via imagehash)

Author: Generated for macOS photo library deduplication
License: MIT
"""

import argparse
import csv
import hashlib
import logging
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("ERROR: Pillow or imagehash not installed. Run: pip install pillow imagehash", file=sys.stderr)
    sys.exit(1)

# Optional HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_REF = "/Volumes/Thunder Drive/iPhotos"
DEFAULT_TARGET = "/Volumes/Thunder Drive/Photos"
DEFAULT_OUTPUT = "/Volumes/Thunder Drive/Unique_to_Import"
DEFAULT_CHUNK_SIZE = 256 * 1024  # 256KB
DEFAULT_PHASH_THRESHOLD = 8
DEFAULT_WORKERS = min(8, os.cpu_count() or 4)

# Mac metadata files/dirs to ignore
IGNORED_FILES = {".DS_Store"}
IGNORED_PREFIXES = ("._",)
IGNORED_DIRS = {".Trash", ".Spotlight-V100", ".fseventsd"}

# Image extensions for perceptual hashing
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Represents a file with its metadata."""
    path: Path
    size: int
    blake2b_hash: Optional[str] = None
    phash: Optional[str] = None


@dataclass
class ReferenceIndex:
    """Index of reference files for deduplication."""
    # size_bytes -> list of FileInfo
    size_buckets: dict = field(default_factory=lambda: defaultdict(list))
    # blake2b_hash -> list of paths
    hash_to_paths: dict = field(default_factory=lambda: defaultdict(list))
    # phash_string -> list of (path, phash_obj) for near-match comparison
    phash_index: dict = field(default_factory=lambda: defaultdict(list))
    total_files: int = 0


@dataclass
class ProcessingResult:
    """Result of processing a single target file."""
    target_path: str
    action: str  # copied, exact_dup, phash_dup, skipped
    match_path: str = ""
    match_type: str = ""  # exact, phash, none
    distance_or_hash: str = ""
    reason: str = ""


@dataclass
class Stats:
    """Statistics for the deduplication run."""
    total_scanned: int = 0
    copied: int = 0
    exact_duplicates: int = 0
    phash_duplicates: int = 0
    skipped: int = 0
    errors: int = 0


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger("dedupe_copy")
    logger.setLevel(logging.DEBUG)

    # File handler - detailed
    log_file = output_dir / "comparison_log.txt"
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler - summary only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def should_ignore_file(path: Path) -> bool:
    """Check if a file should be ignored based on Mac metadata patterns."""
    name = path.name

    # Check ignored filenames
    if name in IGNORED_FILES:
        return True

    # Check ignored prefixes
    if any(name.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True

    return False


def should_ignore_dir(path: Path) -> bool:
    """Check if a directory should be ignored."""
    return path.name in IGNORED_DIRS


def is_image_file(path: Path) -> bool:
    """Check if file is an image that supports perceptual hashing."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def compute_blake2b(file_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute blake2b hash of a file in chunks."""
    hasher = hashlib.blake2b()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to hash {file_path}: {e}")


def compute_phash(file_path: Path) -> Optional[imagehash.ImageHash]:
    """Compute perceptual hash for an image file."""
    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary (handles various color modes)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            return imagehash.phash(img)
    except Exception:
        # Silently skip files that can't be processed as images
        return None


def collect_files(root_dir: Path, logger: logging.Logger) -> list[Path]:
    """
    Recursively collect all valid files from a directory.

    Symlinks to files are SKIPPED (not resolved) for safety and simplicity.
    This avoids potential issues with circular symlinks or links pointing
    outside the intended directory tree.
    """
    files = []

    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        current = Path(dirpath)

        # Filter out ignored directories (modifies in-place to prevent descent)
        dirnames[:] = [d for d in dirnames if not should_ignore_dir(current / d)]

        for filename in filenames:
            file_path = current / filename

            # Skip ignored files
            if should_ignore_file(file_path):
                continue

            # Skip symlinks (documented choice: we skip rather than resolve)
            if file_path.is_symlink():
                logger.debug(f"Skipping symlink: {file_path}")
                continue

            # Skip if not a regular file
            if not file_path.is_file():
                continue

            files.append(file_path)

    return files


def get_relative_path(file_path: Path, base_dir: Path) -> Path:
    """Get the relative path of a file from a base directory."""
    return file_path.relative_to(base_dir)


def safe_copy(src: Path, dst: Path, logger: logging.Logger) -> Path:
    """
    Safely copy a file, handling collisions with numeric suffixes.

    Returns the actual destination path used.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    # Check if existing file is content-identical
    try:
        if dst.stat().st_size == src.stat().st_size:
            src_hash = compute_blake2b(src)
            dst_hash = compute_blake2b(dst)
            if src_hash == dst_hash:
                logger.debug(f"Destination already exists with identical content: {dst}")
                return dst
    except Exception:
        pass

    # Generate collision-safe name
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    counter = 1

    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_dst = parent / new_name
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            logger.debug(f"Collision rename: {dst.name} -> {new_name}")
            return new_dst
        counter += 1
        if counter > 10000:
            raise RuntimeError(f"Too many collisions for {dst}")


# -----------------------------------------------------------------------------
# Indexing Functions
# -----------------------------------------------------------------------------

def index_reference_pass_a(
    ref_files: list[Path],
    chunk_size: int,
    workers: int,
    logger: logging.Logger
) -> ReferenceIndex:
    """
    PASS A: Build reference index with size buckets and blake2b hashes.
    """
    index = ReferenceIndex()
    index.total_files = len(ref_files)

    def process_file(file_path: Path) -> Optional[FileInfo]:
        try:
            size = file_path.stat().st_size
            file_hash = compute_blake2b(file_path, chunk_size)
            return FileInfo(path=file_path, size=size, blake2b_hash=file_hash)
        except Exception as e:
            logger.debug(f"Error indexing {file_path}: {e}")
            return None

    logger.info("PASS A: Indexing reference directory (byte hashes)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, f): f for f in ref_files}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Indexing Reference", unit="files"):
            result = future.result()
            if result:
                index.size_buckets[result.size].append(result)
                index.hash_to_paths[result.blake2b_hash].append(result.path)

    logger.info(f"Reference index: {index.total_files} files in {len(index.size_buckets)} size buckets")
    return index


def index_reference_pass_b(
    ref_files: list[Path],
    workers: int,
    logger: logging.Logger
) -> dict:
    """
    PASS B: Build perceptual hash index for reference images.
    Returns dict: phash_string -> list of (path, phash_obj)
    """
    phash_index = defaultdict(list)
    image_files = [f for f in ref_files if is_image_file(f)]

    if not image_files:
        return phash_index

    def process_image(file_path: Path) -> Optional[tuple]:
        phash = compute_phash(file_path)
        if phash is not None:
            return (file_path, phash)
        return None

    logger.info("PASS B: Building perceptual hash index for reference images...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_image, f): f for f in image_files}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="pHash Reference", unit="images"):
            result = future.result()
            if result:
                path, phash = result
                phash_index[str(phash)].append((path, phash))

    logger.info(f"pHash index: {len(phash_index)} unique perceptual hashes from {len(image_files)} images")
    return phash_index


# -----------------------------------------------------------------------------
# Deduplication Logic
# -----------------------------------------------------------------------------

def find_phash_match(
    target_phash: imagehash.ImageHash,
    phash_index: dict,
    threshold: int
) -> Optional[tuple[Path, int]]:
    """
    Find the best perceptual hash match within threshold.
    Returns (matching_path, distance) or None.
    """
    best_match = None
    best_distance = threshold + 1

    for phash_str, entries in phash_index.items():
        for ref_path, ref_phash in entries:
            distance = target_phash - ref_phash
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_match = (ref_path, distance)
                if distance == 0:
                    return best_match  # Perfect match, stop searching

    return best_match


def process_target_file(
    target_file: Path,
    target_dir: Path,
    output_dir: Path,
    ref_index: ReferenceIndex,
    phash_index: dict,
    phash_threshold: int,
    enable_phash: bool,
    chunk_size: int,
    dry_run: bool,
    logger: logging.Logger
) -> ProcessingResult:
    """
    Process a single target file through the deduplication pipeline.
    """
    target_str = str(target_file)

    try:
        # Get file size
        try:
            size = target_file.stat().st_size
        except Exception as e:
            return ProcessingResult(
                target_path=target_str,
                action="skipped",
                reason=f"Cannot stat file: {e}"
            )

        # PASS A: Check for exact byte-identical duplicates
        target_hash = None
        is_byte_unique = True

        if size in ref_index.size_buckets:
            # Size match exists, compute hash for comparison
            target_hash = compute_blake2b(target_file, chunk_size)

            if target_hash in ref_index.hash_to_paths:
                # Exact duplicate found
                match_path = ref_index.hash_to_paths[target_hash][0]
                return ProcessingResult(
                    target_path=target_str,
                    action="exact_dup",
                    match_path=str(match_path),
                    match_type="exact",
                    distance_or_hash=target_hash[:16],
                    reason="Byte-identical match in reference"
                )

            is_byte_unique = False  # Same size but different hash

        # PASS B: Perceptual hash check for images (only if byte-unique or size-match-but-different)
        if enable_phash and is_image_file(target_file) and phash_index:
            target_phash = compute_phash(target_file)

            if target_phash is not None:
                match = find_phash_match(target_phash, phash_index, phash_threshold)

                if match:
                    match_path, distance = match
                    return ProcessingResult(
                        target_path=target_str,
                        action="phash_dup",
                        match_path=str(match_path),
                        match_type="phash",
                        distance_or_hash=str(distance),
                        reason=f"Perceptual match (distance={distance})"
                    )

        # File is unique - copy it
        rel_path = get_relative_path(target_file, target_dir)
        dest_path = output_dir / rel_path

        if dry_run:
            return ProcessingResult(
                target_path=target_str,
                action="copied",
                match_type="none",
                distance_or_hash=target_hash[:16] if target_hash else "",
                reason=f"[DRY-RUN] Would copy to {dest_path}"
            )

        actual_dest = safe_copy(target_file, dest_path, logger)
        return ProcessingResult(
            target_path=target_str,
            action="copied",
            match_type="none",
            distance_or_hash=target_hash[:16] if target_hash else "",
            reason=f"Copied to {actual_dest}"
        )

    except Exception as e:
        logger.debug(f"Error processing {target_file}: {e}")
        return ProcessingResult(
            target_path=target_str,
            action="skipped",
            reason=f"Error: {e}"
        )


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def run_deduplication(
    ref_dir: Path,
    target_dir: Path,
    output_dir: Path,
    workers: int,
    chunk_size: int,
    phash_threshold: int,
    enable_phash: bool,
    dry_run: bool,
    extensions: Optional[set[str]],
    logger: logging.Logger
) -> tuple[Stats, list[ProcessingResult]]:
    """
    Main deduplication workflow.
    """
    stats = Stats()
    results = []

    # Collect files
    logger.info(f"Scanning reference directory: {ref_dir}")
    ref_files = collect_files(ref_dir, logger)
    logger.info(f"Found {len(ref_files)} files in reference")

    logger.info(f"Scanning target directory: {target_dir}")
    target_files = collect_files(target_dir, logger)
    logger.info(f"Found {len(target_files)} files in target")

    # Filter by extensions if specified
    if extensions:
        target_files = [f for f in target_files if f.suffix.lower() in extensions]
        ref_files = [f for f in ref_files if f.suffix.lower() in extensions]
        logger.info(f"After extension filter: {len(ref_files)} ref, {len(target_files)} target")

    stats.total_scanned = len(target_files)

    if not target_files:
        logger.info("No target files to process")
        return stats, results

    # PASS A: Build reference index
    ref_index = index_reference_pass_a(ref_files, chunk_size, workers, logger)

    # PASS B: Build perceptual hash index (if enabled)
    phash_index = {}
    if enable_phash:
        phash_index = index_reference_pass_b(ref_files, workers, logger)

    # Process target files
    logger.info(f"Processing {len(target_files)} target files...")

    if dry_run:
        logger.info("*** DRY-RUN MODE - No files will be copied ***")

    def process_wrapper(f):
        return process_target_file(
            f, target_dir, output_dir, ref_index, phash_index,
            phash_threshold, enable_phash, chunk_size, dry_run, logger
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_wrapper, f): f for f in target_files}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Processing Target", unit="files"):
            result = future.result()
            results.append(result)

            # Update stats
            if result.action == "copied":
                stats.copied += 1
            elif result.action == "exact_dup":
                stats.exact_duplicates += 1
            elif result.action == "phash_dup":
                stats.phash_duplicates += 1
            elif result.action == "skipped":
                stats.skipped += 1

    return stats, results


def write_results_csv(results: list[ProcessingResult], output_dir: Path):
    """Write results to CSV file."""
    csv_path = output_dir / "results.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_path", "action", "match_path", "match_type",
            "distance_or_hash", "reason"
        ])

        for r in results:
            writer.writerow([
                r.target_path, r.action, r.match_path, r.match_type,
                r.distance_or_hash, r.reason
            ])


def print_summary(stats: Stats, logger: logging.Logger, dry_run: bool):
    """Print final summary statistics."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY" + (" (DRY-RUN)" if dry_run else ""))
    logger.info("=" * 60)
    logger.info(f"Total scanned:      {stats.total_scanned:,}")
    logger.info(f"Copied:             {stats.copied:,}")
    logger.info(f"Exact duplicates:   {stats.exact_duplicates:,}")
    logger.info(f"pHash duplicates:   {stats.phash_duplicates:,}")
    logger.info(f"Skipped/errors:     {stats.skipped:,}")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate and copy unique files from Target to Output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run with defaults
  python dedupe_copy.py --dry-run

  # Full run with custom paths
  python dedupe_copy.py --ref /path/to/reference --target /path/to/target --out /path/to/output

  # Disable perceptual hashing (faster, byte-only comparison)
  python dedupe_copy.py --disable-phash

  # Custom pHash threshold (lower = stricter matching)
  python dedupe_copy.py --phash-threshold 5

  # Process only JPEGs
  python dedupe_copy.py --extensions .jpg,.jpeg
        """
    )

    parser.add_argument(
        "--ref", type=Path, default=Path(DEFAULT_REF),
        help=f"Reference directory (default: {DEFAULT_REF})"
    )
    parser.add_argument(
        "--target", type=Path, default=Path(DEFAULT_TARGET),
        help=f"Target directory to scan (default: {DEFAULT_TARGET})"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(DEFAULT_OUTPUT),
        help=f"Output directory for unique files (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of worker threads (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for hashing in bytes (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--phash-threshold", type=int, default=DEFAULT_PHASH_THRESHOLD,
        help=f"Perceptual hash distance threshold (default: {DEFAULT_PHASH_THRESHOLD})"
    )
    parser.add_argument(
        "--disable-phash", action="store_true",
        help="Disable perceptual hash comparison (PASS B)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be copied without actually copying"
    )
    parser.add_argument(
        "--extensions", type=str, default=None,
        help="Comma-separated list of extensions to process (e.g., .jpg,.png)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate directories
    if not args.ref.exists():
        print(f"ERROR: Reference directory does not exist: {args.ref}", file=sys.stderr)
        sys.exit(1)

    if not args.target.exists():
        print(f"ERROR: Target directory does not exist: {args.target}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.out)

    # Log configuration
    logger.info("=" * 60)
    logger.info("PHOTO DEDUPLICATION TOOL")
    logger.info("=" * 60)
    logger.info(f"Reference:     {args.ref}")
    logger.info(f"Target:        {args.target}")
    logger.info(f"Output:        {args.out}")
    logger.info(f"Workers:       {args.workers}")
    logger.info(f"Chunk size:    {args.chunk_size:,} bytes")
    logger.info(f"pHash enabled: {not args.disable_phash}")
    if not args.disable_phash:
        logger.info(f"pHash thresh:  {args.phash_threshold}")
    logger.info(f"HEIC support:  {HEIC_SUPPORTED}")
    logger.info(f"Dry-run:       {args.dry_run}")
    logger.info(f"Started:       {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Parse extensions filter
    extensions = None
    if args.extensions:
        extensions = set(ext.strip().lower() if ext.startswith('.') else f".{ext.strip().lower()}"
                        for ext in args.extensions.split(','))
        logger.info(f"Extension filter: {extensions}")

    # Run deduplication
    try:
        stats, results = run_deduplication(
            ref_dir=args.ref,
            target_dir=args.target,
            output_dir=args.out,
            workers=args.workers,
            chunk_size=args.chunk_size,
            phash_threshold=args.phash_threshold,
            enable_phash=not args.disable_phash,
            dry_run=args.dry_run,
            extensions=extensions,
            logger=logger
        )

        # Write results CSV
        write_results_csv(results, args.out)
        logger.info(f"Results written to: {args.out / 'results.csv'}")

        # Print summary
        print_summary(stats, logger, args.dry_run)

        logger.info(f"Completed: {datetime.now().isoformat()}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
