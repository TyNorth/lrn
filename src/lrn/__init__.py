from lrn.node import Node, assign_roles, ROLE_NAMES
from lrn.spring import Spring, SPRING_PROMOTION_THRESHOLDS
from lrn.lattice import LatticeNN, FLAG_PINNED, FLAG_DAMPENED, FLAG_EPHEMERAL
from lrn.propagate import propagate, _is_stable, propagate_with_negative
from lrn.corpus import CorpusExpander
from lrn.training import (
    add_sentence, add_negative_sentence, add_identity_anchor, train_corpus,
    train_language_native, train_all_languages_native, train_mixed_language_native,
    train_language_to_grade,
)
from lrn.generate import generate, generate_sequence, _score_candidates
from lrn.code_tokenizer import tokenize_code_line, assign_code_roles as assign_code_roles_fn, SUPPORTED_LANGUAGES
from lrn.code_training import (
    add_code_file, initialize_scope_axis, promote_code_springs,
    train_code_grammar, train_all_languages, train_language,
    train_native_code, train_all_languages_native,
    PYTHON_PATTERNS, RUST_PATTERNS, JAVASCRIPT_PATTERNS, GO_PATTERNS,
    RUBY_PATTERNS, COBOL_PATTERNS, ZIG_PATTERNS,
    ALL_LANGUAGE_PATTERNS, ALL_LOGIC_PATTERNS,
    NATIVE_CODE_SAMPLES,
)
from lrn.native_tokenize import (
    learn_from_code, discover_tokens, native_tokenize,
    CHAR_NGRAM_SIZES as CODE_CHAR_NGRAM_SIZES, TAU_THRESHOLD as CODE_TAU_THRESHOLD,
)
from lrn.natural_tokenize import (
    learn_from_text, discover_words, native_tokenize as tokenize_text,
    CHAR_NGRAM_SIZES as LANG_CHAR_NGRAM_SIZES, TAU_THRESHOLD as LANG_TAU_THRESHOLD,
    learn_mixed_language, discover_language_markers,
)
from lrn.language_corpus import (
    NATIVE_LANGUAGE_SAMPLES, GRADE_THRESHOLDS, get_grade,
    get_language_samples, get_mixed_samples,
)
from lrn.code_logic import (
    initialize_logic_nodes, compare_to_springs, add_boolean_condition,
    logic_residual, solve_condition, add_logic_pattern, train_logic_patterns,
    BOOLEAN_TRUE, BOOLEAN_FALSE,
)
from lrn.code_generate import (
    generate_code, validate_syntax, generate_code_sequence, test_generation,
    CODE_E_THRESHOLD, CODE_CANDIDATE_MULTIPLIER,
)

__all__ = [
    "Node", "Spring", "LatticeNN",
    "assign_roles", "ROLE_NAMES", "SPRING_PROMOTION_THRESHOLDS",
    "propagate", "propagate_with_negative", "_is_stable",
    "FLAG_PINNED", "FLAG_DAMPENED", "FLAG_EPHEMERAL",
    "CorpusExpander", "add_sentence", "add_negative_sentence",
    "add_identity_anchor", "train_corpus",
    "train_language_native", "train_all_languages_native", "train_mixed_language_native",
    "train_language_to_grade",
    "generate", "generate_sequence", "_score_candidates",
    "tokenize_code_line", "assign_code_roles_fn", "SUPPORTED_LANGUAGES",
    "add_code_file", "initialize_scope_axis", "promote_code_springs",
    "train_code_grammar", "train_all_languages", "train_language",
    "train_native_code", "train_all_languages_native",
    "PYTHON_PATTERNS", "RUST_PATTERNS", "JAVASCRIPT_PATTERNS", "GO_PATTERNS",
    "RUBY_PATTERNS", "COBOL_PATTERNS", "ZIG_PATTERNS",
    "ALL_LANGUAGE_PATTERNS", "ALL_LOGIC_PATTERNS",
    "NATIVE_CODE_SAMPLES",
    "learn_from_code", "discover_tokens", "native_tokenize",
    "CHAR_NGRAM_SIZES", "TAU_THRESHOLD",
    "learn_from_text", "discover_words", "tokenize_text",
    "LANG_CHAR_NGRAM_SIZES", "LANG_TAU_THRESHOLD",
    "learn_mixed_language", "discover_language_markers",
    "NATIVE_LANGUAGE_SAMPLES", "GRADE_THRESHOLDS", "get_grade",
    "get_language_samples", "get_mixed_samples",
    "generate_code", "validate_syntax", "generate_code_sequence", "test_generation",
    "CODE_E_THRESHOLD", "CODE_CANDIDATE_MULTIPLIER",
    "initialize_logic_nodes", "compare_to_springs", "add_boolean_condition",
    "logic_residual", "solve_condition", "add_logic_pattern", "train_logic_patterns",
    "BOOLEAN_TRUE", "BOOLEAN_FALSE",
]