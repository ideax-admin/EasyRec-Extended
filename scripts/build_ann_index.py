#!/usr/bin/env python3
"""CLI script to build a Faiss ANN index from item embedding files.

Usage examples::

    # From a .npy file (shape: (n_items, dim))
    python scripts/build_ann_index.py \\
        --embedding_file exports/item_embeddings.npy \\
        --output_path data/item_index \\
        --dimension 64

    # From a CSV file (first column = item_id, remaining columns = embedding)
    python scripts/build_ann_index.py \\
        --embedding_file exports/item_embeddings.csv \\
        --output_path data/item_index \\
        --dimension 64 \\
        --index_type Flat
"""
import argparse
import logging
import os
import sys

import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _load_npy(path: str):
    """Load embeddings from a numpy .npy file.

    The file must contain a 2-D float array of shape ``(n, dim)``.
    Item IDs are generated as ``"item_0"``, ``"item_1"``, …

    Returns:
        Tuple ``(embeddings, ids)``.
    """
    embeddings = np.load(path).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {embeddings.shape}")
    ids = [f"item_{i}" for i in range(len(embeddings))]
    return embeddings, ids


def _load_csv(path: str):
    """Load embeddings from a CSV file.

    The first column must be the string item ID; the remaining columns are
    the embedding values.

    Returns:
        Tuple ``(embeddings, ids)``.
    """
    import csv

    ids = []
    rows = []
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            ids.append(row[0])
            rows.append([float(v) for v in row[1:]])
    embeddings = np.array(rows, dtype=np.float32)
    return embeddings, ids


def main():
    parser = argparse.ArgumentParser(
        description='Build a Faiss ANN index from item embedding files.'
    )
    parser.add_argument(
        '--embedding_file',
        required=True,
        help='Path to the embedding file (.npy or .csv).',
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Base output path (without extension) for the saved index.',
    )
    parser.add_argument(
        '--dimension',
        type=int,
        required=True,
        help='Dimensionality of the embedding vectors.',
    )
    parser.add_argument(
        '--index_type',
        default='IVFFlat',
        choices=['Flat', 'IVFFlat'],
        help='Faiss index type (default: IVFFlat).',
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=100,
        help='Number of IVF cells (only for IVFFlat index, default: 100).',
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=10,
        help='Number of IVF cells to probe at search time (default: 10).',
    )
    args = parser.parse_args()

    # Load embeddings
    ext = os.path.splitext(args.embedding_file)[1].lower()
    if ext == '.npy':
        embeddings, ids = _load_npy(args.embedding_file)
    elif ext == '.csv':
        embeddings, ids = _load_csv(args.embedding_file)
    else:
        parser.error(f"Unsupported file format: {ext!r}. Use .npy or .csv.")

    logger.info(
        "Loaded %d embeddings of dimension %d from %s",
        len(ids),
        embeddings.shape[1],
        args.embedding_file,
    )

    if embeddings.shape[1] != args.dimension:
        logger.warning(
            "Detected dimension %d does not match --dimension %d; "
            "using detected dimension.",
            embeddings.shape[1],
            args.dimension,
        )
        args.dimension = embeddings.shape[1]

    # Build index
    from easyrec_extended.ann.faiss_index import FaissANNIndex

    index = FaissANNIndex(
        dimension=args.dimension,
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    index.build(embeddings, ids)
    index.save(args.output_path)
    logger.info("Index saved to %s(.faiss/.meta)", args.output_path)


if __name__ == '__main__':
    main()
