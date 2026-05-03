"""
LRN Level Corpora Registry
"""

AVAILABLE_LEVELS = ["prek", "kindergarten", "first_grade", "second_grade", "third_grade", "fourth_grade", "fifth_grade", "sixth_grade", "seventh_grade", "eighth_grade", "ninth_grade", "tenth_grade", "eleventh_grade", "twelfth_grade", "college"]

def get_corpus(level):
    if level == "prek":
        from lrn.corpora.prek import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "kindergarten":
        from lrn.corpora.kindergarten import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "first_grade":
        from lrn.corpora.first_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "second_grade":
        from lrn.corpora.second_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "third_grade":
        from lrn.corpora.third_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "fourth_grade":
        from lrn.corpora.fourth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "fifth_grade":
        from lrn.corpora.fifth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "sixth_grade":
        from lrn.corpora.sixth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "seventh_grade":
        from lrn.corpora.seventh_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "eighth_grade":
        from lrn.corpora.eighth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "ninth_grade":
        from lrn.corpora.ninth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "tenth_grade":
        from lrn.corpora.tenth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "eleventh_grade":
        from lrn.corpora.eleventh_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "twelfth_grade":
        from lrn.corpora.twelfth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    elif level == "college":
        from lrn.corpora.college import ORIGINAL_CORPUS, VARIED_EXAMPLES, FULL_CORPUS
        return FULL_CORPUS
    raise ValueError(f"Unknown level: {level}")

def get_corpus_info(level):
    if level == "prek":
        from lrn.corpora.prek import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "kindergarten":
        from lrn.corpora.kindergarten import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "first_grade":
        from lrn.corpora.first_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "second_grade":
        from lrn.corpora.second_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "third_grade":
        from lrn.corpora.third_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "fourth_grade":
        from lrn.corpora.fourth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "fifth_grade":
        from lrn.corpora.fifth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "sixth_grade":
        from lrn.corpora.sixth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "seventh_grade":
        from lrn.corpora.seventh_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "eighth_grade":
        from lrn.corpora.eighth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "ninth_grade":
        from lrn.corpora.ninth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "tenth_grade":
        from lrn.corpora.tenth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "eleventh_grade":
        from lrn.corpora.eleventh_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "twelfth_grade":
        from lrn.corpora.twelfth_grade import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    elif level == "college":
        from lrn.corpora.college import ORIGINAL_CORPUS, VARIED_EXAMPLES
        return {"original": len(ORIGINAL_CORPUS), "varied": len(VARIED_EXAMPLES), "total": len(ORIGINAL_CORPUS) + len(VARIED_EXAMPLES)}
    raise ValueError(f"Unknown level: {level}")
