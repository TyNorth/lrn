"""
LRN Level Corpora Registry
"""

AVAILABLE_LEVELS = ["prek", "kindergarten"]

def get_corpus(level):
    if level == "prek":
        from lrn.corpora.prek import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "kindergarten":
        from lrn.corpora.kindergarten import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    raise ValueError(f"Unknown level: {level}")

def get_corpus_info(level):
    if level == "prek":
        from lrn.corpora.prek import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {
            "original": len(ORIGINAL_CORPUS),
            "varied": len(VARIED_EXAMPLES),
            "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES),
        }
    elif level == "kindergarten":
        from lrn.corpora.kindergarten import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {
            "original": len(ORIGINAL_CORPUS),
            "varied": len(VARIED_EXAMPLES),
            "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES),
        }
    raise ValueError(f"Unknown level: {level}")
