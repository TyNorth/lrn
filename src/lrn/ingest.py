"""
Book Ingestion - Chunk by paragraph, periodic REM, natural metadata
"""
import sys
import os
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.trainer import optimal_rem, train
from lrn import propagate
from lrn.charts import progress_bar


def parse_metadata(paragraphs):
    """Extract metadata naturally from first few paragraphs."""
    metadata = {}
    first_lines = paragraphs[:3] if len(paragraphs) >= 3 else paragraphs
    
    for line in first_lines:
        line_lower = line.lower()
        if "by " in line_lower:
            parts = line.split("by ")
            if len(parts) > 1:
                metadata["author"] = parts[1].strip()
        if "illustrated" in line_lower:
            parts = line_lower.split("illustrated by ")
            if len(parts) > 1:
                metadata["illustrator"] = parts[1].strip()
    
    if first_lines:
        metadata["title"] = first_lines[0].strip()
    
    return metadata


def chunk_by_paragraph(text):
    """Split text into paragraphs (blank line = paragraph boundary)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def ingest_book(lnn, filepath, level="prek", rem_interval=50, verbose=True):
    """
    Ingest a book into the lattice.
    
    Args:
        lnn: LatticeNN instance
        filepath: path to book text file
        level: training level for learn_type
        rem_interval: REM every N paragraphs
        verbose: print progress
    """
    with open(filepath, "r") as f:
        text = f.read()
    
    paragraphs = chunk_by_paragraph(text)
    metadata = parse_metadata(paragraphs)
    
    if verbose:
        print(f"  Book: {metadata.get('title', 'Unknown')}")
        if "author" in metadata:
            print(f"  Author: {metadata['author']}")
        if "illustrator" in metadata:
            print(f"  Illustrator: {metadata['illustrator']}")
        print(f"  Paragraphs: {len(paragraphs)}")
        print(f"  REM interval: every {rem_interval} paragraphs")
        print()
    
    wake_buffer = []
    rem_count = 0
    para_count = 0
    
    learn_type = "sensory" if level in ("prek",) else "language"
    
    for idx, para in enumerate(paragraphs):
        para_count += 1
        
        # Train paragraph
        from lrn.natural_tokenize import learn_from_text
        from lrn.inference import add_word_nodes
        
        learn_from_text(lnn, para, repetitions=1, learn_type=learn_type)
        add_word_nodes(lnn, [para])
        
        wake_buffer.append(para)
        if len(wake_buffer) > 20:
            wake_buffer = wake_buffer[-20:]
        
        # Periodic REM
        if (idx + 1) % rem_interval == 0:
            optimal_rem(lnn, wake_buffer)
            propagate(lnn, n_steps=3)
            rem_count += 1
        
        if verbose and (idx + 1) % 10 == 0:
            print(progress_bar(idx + 1, len(paragraphs)), end="", flush=True)
    
    # Final REM
    optimal_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(progress_bar(len(paragraphs), len(paragraphs)))
        print(f"  REM cycles: {rem_count}")
        print()
    
    return {
        "metadata": metadata,
        "paragraphs": para_count,
        "rem_cycles": rem_count,
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
    }
