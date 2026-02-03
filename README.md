# Photo Dedupe Copy

A production-grade Python tool for comparing two large photo directories and copying only unique files using a two-pass duplicate detection strategy.

Designed for macOS external drive workflows (e.g., consolidating iPhoto/Photos libraries).

## Features

- **PASS A**: Exact byte-identical duplicate detection using `blake2b` hashing
- **PASS B**: Perceptual hash (pHash) detection for visually similar images
- Preserves directory structure and file metadata
- Multi-threaded for optimal I/O performance on external drives
- Progress bars for all operations
- Detailed logging and CSV output
- Dry-run mode for safe previewing
- HEIC support (optional)

## Requirements

- Python 3.11+
- macOS (optimized for external Thunderbolt drives)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/photo-dedupe.git
cd photo-dedupe
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: HEIC support

For HEIC/HEIF image support (common on iPhones):

```bash
pip install pillow-heif
```

## Usage

### Default paths (external Thunderbolt drive)

```bash
# Dry-run first (recommended)
python dedupe_copy.py --dry-run

# Full run
python dedupe_copy.py
```

Default paths:
- Reference: `/Volumes/Thunder Drive/iPhotos`
- Target: `/Volumes/Thunder Drive/Photos`
- Output: `/Volumes/Thunder Drive/Unique_to_Import`

### Custom paths

```bash
python dedupe_copy.py \
    --ref /path/to/reference \
    --target /path/to/target \
    --out /path/to/output
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--ref` | Reference directory (files to compare against) | `/Volumes/Thunder Drive/iPhotos` |
| `--target` | Target directory (files to check for uniqueness) | `/Volumes/Thunder Drive/Photos` |
| `--out` | Output directory (where unique files are copied) | `/Volumes/Thunder Drive/Unique_to_Import` |
| `--workers` | Number of worker threads | `min(8, cpu_count)` |
| `--chunk-size` | Chunk size for hashing (bytes) | `262144` (256KB) |
| `--phash-threshold` | pHash distance threshold (lower = stricter) | `8` |
| `--disable-phash` | Disable perceptual hashing (PASS B) | `False` |
| `--dry-run` | Preview without copying | `False` |
| `--extensions` | Comma-separated extension filter | All files |

### Examples

```bash
# Process only JPEGs and PNGs
python dedupe_copy.py --extensions .jpg,.jpeg,.png

# Stricter perceptual matching
python dedupe_copy.py --phash-threshold 5

# Byte-only comparison (faster, no perceptual hashing)
python dedupe_copy.py --disable-phash

# More workers for faster processing
python dedupe_copy.py --workers 16
```

## Output

The tool creates the following in the output directory:

### 1. Copied files
Unique files from Target are copied preserving their relative directory structure.

### 2. `comparison_log.txt`
Human-readable log with timestamps and summary.

### 3. `results.csv`
Detailed results with columns:
- `target_path`: Source file path
- `action`: `copied`, `exact_dup`, `phash_dup`, or `skipped`
- `match_path`: Path of matching reference file (if duplicate)
- `match_type`: `exact`, `phash`, or `none`
- `distance_or_hash`: Hash prefix or pHash distance
- `reason`: Human-readable explanation

## How It Works

### PASS A: Byte-Identical Detection

1. Index all reference files by size (creates size buckets)
2. Compute `blake2b` hash for each reference file
3. For each target file:
   - If size not in any bucket → unique (skip to PASS B or copy)
   - If size matches → compute hash and compare
   - If hash matches → exact duplicate (skip)

### PASS B: Perceptual Hash Detection

1. Build pHash index for all reference images
2. For target images that passed PASS A:
   - Compute perceptual hash
   - Find closest match in reference index
   - If distance ≤ threshold → perceptual duplicate (skip)

Supported image formats: `jpg`, `jpeg`, `png`, `heic`, `webp`, `tif`, `tiff`

## Limitations

1. **PASS A only catches byte-identical duplicates**
   - Resized, re-encoded, or cropped versions are NOT detected
   - Different file formats of the same image are NOT detected

2. **PASS B catches resized/re-encoded near-duplicates for images only**
   - Threshold of 8 may miss heavily edited images
   - Very different crops may not match
   - Small thumbnails may false-positive

3. **Videos are NOT content-deduplicated**
   - Only byte-identical video duplicates are detected
   - No perceptual hashing for video content

4. **Symlinks are skipped**
   - Symlinks to files are ignored for safety
   - This prevents circular reference issues

5. **Performance depends on I/O**
   - External drive speed is usually the bottleneck
   - Thunderbolt drives recommended for large libraries

## File Handling

### Ignored files (Mac metadata)
- `.DS_Store`
- Files starting with `._`
- Contents of `.Trash`, `.Spotlight-V100`, `.fseventsd`

### Collision handling
- If destination file exists with identical content → skip
- If destination file exists with different content → rename with numeric suffix (e.g., `photo_1.jpg`)

## Safety

- **Never deletes or modifies source files**
- Uses `shutil.copy2` to preserve metadata
- Dry-run mode for safe previewing
- Creates output directory if missing

## Git Setup (for contributors)

```bash
# Initialize repository
git init

# Add files
git add dedupe_copy.py README.md requirements.txt .gitignore

# Initial commit
git commit -m "Initial commit: photo deduplication tool"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/photo-dedupe.git

# Push to GitHub
git push -u origin main
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests with sample data
5. Submit a pull request
