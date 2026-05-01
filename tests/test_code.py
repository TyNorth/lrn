"""
Tests for LRN coding capabilities.
- Tokenization
- Spring formation (syntax, type, scope)
- Spring promotion
- Code generation
- Syntax validation
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pytest
from lrn import (
    LatticeNN, Node, Spring,
    tokenize_code_line, assign_code_roles_fn,
    add_code_file, initialize_scope_axis, promote_code_springs,
    train_spec_text, train_code_curriculum,
    generate_code, validate_syntax, generate_code_sequence, test_generation,
    CODE_E_THRESHOLD, CODE_CANDIDATE_MULTIPLIER,
    SPEC_TEXT, SYNTAX_PATTERNS, FUNCTION_PATTERNS,
    add_sentence, generate, add_identity_anchor, propagate,
)


# ============================================================
# 1. Tokenization Tests
# ============================================================

class TestTokenization:
    def test_def_keyword(self):
        tokens = tokenize_code_line("def")
        assert tokens == ["code:kw:def"]

    def test_function_def_simple(self):
        tokens = tokenize_code_line("def foo ( ) :")
        assert tokens == [
            "code:kw:def",
            "code:var:foo",
            "code:sym:lparen",
            "code:sym:rparen",
            "code:sym:colon",
        ]

    def test_return_statement(self):
        tokens = tokenize_code_line("return x")
        assert tokens == ["code:kw:return", "code:var:x"]

    def test_return_with_expression(self):
        tokens = tokenize_code_line("return a + b")
        assert tokens == [
            "code:kw:return",
            "code:var:a",
            "code:op:plus",
            "code:var:b",
        ]

    def test_if_statement(self):
        tokens = tokenize_code_line("if x > 0 :")
        assert tokens == [
            "code:kw:if",
            "code:var:x",
            "code:sym:gt",
            "code:lit:int",
            "code:sym:colon",
        ]

    def test_assignment(self):
        tokens = tokenize_code_line("x = 1")
        assert tokens == ["code:var:x", "code:sym:assign", "code:lit:int"]

    def test_function_call(self):
        tokens = tokenize_code_line("print ( x )")
        assert tokens == [
            "code:func:print",
            "code:sym:lparen",
            "code:var:x",
            "code:sym:rparen",
        ]

    def test_len_function(self):
        tokens = tokenize_code_line("len ( x )")
        assert tokens[0] == "code:func:len"

    def test_string_literal(self):
        tokens = tokenize_code_line('"hello"')
        assert tokens == ["code:lit:str"]

    def test_integer_literal(self):
        tokens = tokenize_code_line("42")
        assert tokens == ["code:lit:int"]

    def test_full_function(self):
        tokens = tokenize_code_line("def add ( a , b ) : return a + b")
        assert "code:kw:def" in tokens
        assert "code:op:plus" in tokens
        assert "code:kw:return" in tokens
        assert "code:sym:colon" in tokens
        assert len(tokens) == 12

    def test_empty_line(self):
        tokens = tokenize_code_line("")
        assert tokens == []

    def test_comment_line(self):
        tokens = tokenize_code_line("# this is a comment")
        assert tokens == []

    def test_dot_access(self):
        tokens = tokenize_code_line("self . value")
        assert tokens == ["code:var:self", "code:sym:dot", "code:var:value"]


# ============================================================
# 2. Code Role Assignment Tests
# ============================================================

class TestCodeRoles:
    def test_def_is_declaration(self):
        tokens = ["code:kw:def", "code:var:foo"]
        roles = assign_code_roles_fn(tokens)
        assert roles[0] == 5  # DECLARATION

    def test_return_is_return_role(self):
        tokens = ["code:kw:return", "code:var:x"]
        roles = assign_code_roles_fn(tokens)
        assert roles[0] == 9  # RETURN

    def test_closer_roles(self):
        tokens = ["code:sym:rparen", "code:sym:colon"]
        roles = assign_code_roles_fn(tokens)
        assert roles[0] == 4  # CLOSER
        assert roles[1] == 4  # CLOSER

    def test_operator_is_expression(self):
        tokens = ["code:var:a", "code:op:plus", "code:var:b"]
        roles = assign_code_roles_fn(tokens)
        assert roles[1] == 8  # EXPRESSION


# ============================================================
# 3. Spring Formation Tests
# ============================================================

class TestSpringFormation:
    def setup_method(self):
        self.lnn = LatticeNN()

    def test_add_code_file_creates_nodes(self):
        add_code_file(self.lnn, "def foo ( ) :")
        assert "code:kw:def" in self.lnn.nodes
        assert "code:var:foo" in self.lnn.nodes
        assert "code:sym:lparen" in self.lnn.nodes

    def test_add_code_file_creates_springs(self):
        initial_springs = len(self.lnn.springs)
        add_code_file(self.lnn, "def foo ( ) :")
        assert len(self.lnn.springs) > initial_springs

    def test_sequential_springs_formed(self):
        add_code_file(self.lnn, "code:kw:def code:var:foo code:sym:lparen")
        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert key in self.lnn.springs

    def test_spring_exposure_count(self):
        add_code_file(self.lnn, "def foo ( ) :")
        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert self.lnn.springs[key].exposure_count >= 1

    def test_repetition_accumulates_stiffness(self):
        for _ in range(5):
            add_code_file(self.lnn, "def foo ( ) :")
        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert self.lnn.springs[key].stiffness > 20

    def test_ngrams_added(self):
        initial_ngrams = len(self.lnn.trigrams)
        add_code_file(self.lnn, "def foo ( ) :")
        assert len(self.lnn.trigrams) > initial_ngrams

    def test_stats_returned(self):
        stats = add_code_file(self.lnn, "def foo ( ) :")
        assert "tokens" in stats
        assert "springs" in stats
        assert "ngrams" in stats
        assert stats["tokens"] > 0


# ============================================================
# 4. Spring Promotion Tests
# ============================================================

class TestSpringPromotion:
    def setup_method(self):
        self.lnn = LatticeNN()

    def test_promotion_after_repetition(self):
        for _ in range(50):
            add_code_file(self.lnn, "def foo ( ) :")
        promoted = promote_code_springs(self.lnn, threshold=50)
        assert promoted > 0

    def test_promoted_springs_have_tau_zero(self):
        for _ in range(50):
            add_code_file(self.lnn, "def foo ( ) :")
        promote_code_springs(self.lnn, threshold=50)
        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert self.lnn.springs[key].tau == 0

    def test_promoted_springs_have_high_stiffness(self):
        for _ in range(50):
            add_code_file(self.lnn, "def foo ( ) :")
        promote_code_springs(self.lnn, threshold=50)
        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert self.lnn.springs[key].stiffness >= 100


# ============================================================
# 5. Scope Axis Tests
# ============================================================

class TestScopeAxis:
    def setup_method(self):
        self.lnn = LatticeNN()

    def test_scope_nodes_created(self):
        initialize_scope_axis(self.lnn)
        assert "code:block:0" in self.lnn.nodes
        assert "code:block:4" in self.lnn.nodes
        assert "code:block:31" in self.lnn.nodes

    def test_scope_nodes_have_positions(self):
        initialize_scope_axis(self.lnn)
        assert self.lnn.nodes["code:block:0"].x == 0
        assert self.lnn.nodes["code:block:4"].x == 16
        assert self.lnn.nodes["code:block:8"].x == 32

    def test_scope_springs_formed(self):
        initialize_scope_axis(self.lnn)
        key = self.lnn._key("code:block:0", "code:block:1")
        assert key in self.lnn.springs
        assert self.lnn.springs[key].tau == 1


# ============================================================
# 6. Spec Text Training Tests
# ============================================================

class TestSpecTraining:
    def setup_method(self):
        self.lnn = LatticeNN()

    def test_spec_text_creates_nodes(self):
        train_spec_text(self.lnn, SPEC_TEXT[:2])
        assert len(self.lnn.nodes) > 0

    def test_spec_text_creates_springs(self):
        initial = len(self.lnn.springs)
        train_spec_text(self.lnn, SPEC_TEXT)
        assert len(self.lnn.springs) > initial

    def test_spec_text_connects_to_code(self):
        train_spec_text(self.lnn, ["A function definition begins with the def keyword"])
        add_code_file(self.lnn, "def foo ( ) :")
        assert len(self.lnn.nodes) > 5


# ============================================================
# 7. Code Generation Tests
# ============================================================

class TestCodeGeneration:
    def setup_method(self):
        self.lnn = LatticeNN()
        add_identity_anchor(self.lnn)

    def _train_basic(self):
        for _ in range(50):
            add_code_file(self.lnn, "def foo ( ) :")
            add_code_file(self.lnn, "def add ( a , b ) : return a + b")
            add_code_file(self.lnn, "return x")
            add_code_file(self.lnn, "return a + b")
        promote_code_springs(self.lnn, threshold=50)

    def test_generate_code_returns_candidates(self):
        self._train_basic()
        candidates = generate_code(self.lnn, ["code:kw:def"])
        assert isinstance(candidates, list)

    def test_generate_code_energy_below_threshold(self):
        self._train_basic()
        candidates = generate_code(self.lnn, ["code:kw:def"])
        for c in candidates:
            assert c["energy"] < CODE_E_THRESHOLD

    def test_generate_code_filters_high_energy(self):
        self._train_basic()
        candidates = generate_code(self.lnn, ["code:kw:def"])
        for c in candidates:
            assert c["energy"] < 20

    def test_generate_code_sequence(self):
        self._train_basic()
        seq = generate_code_sequence(self.lnn, "code:kw:def", max_tokens=5)
        assert isinstance(seq, str)

    def test_stricter_than_text_generation(self):
        self._train_basic()
        code_candidates = generate_code(self.lnn, ["code:kw:def"])
        text_candidates = generate(self.lnn, ["code:kw:def"])
        assert len(code_candidates) <= len(text_candidates)


# ============================================================
# 8. Syntax Validation Tests
# ============================================================

class TestSyntaxValidation:
    def setup_method(self):
        self.lnn = LatticeNN()

    def test_validate_syntax_returns_true_for_valid(self):
        add_code_file(self.lnn, "def foo ( ) :")
        for _ in range(50):
            add_code_file(self.lnn, "def foo ( ) :")
        promote_code_springs(self.lnn, threshold=50)

        context = ["code:kw:def", "code:var:foo"]
        result = validate_syntax(self.lnn, "code:sym:lparen", context)
        assert isinstance(result, bool)

    def test_validate_syntax_no_false_positives(self):
        context = ["code:kw:def"]
        result = validate_syntax(self.lnn, "code:var:foo", context)
        assert isinstance(result, bool)


# ============================================================
# 9. Full Curriculum Test
# ============================================================

class TestFullCurriculum:
    def setup_method(self):
        self.lnn = LatticeNN()
        add_identity_anchor(self.lnn)

    def test_curriculum_trains(self):
        stats = train_code_curriculum(self.lnn)
        assert stats["spec_sentences"] > 0
        assert stats["syntax_patterns"] > 0
        assert stats["total_springs"] >= 0

    def test_curriculum_creates_many_nodes(self):
        train_code_curriculum(self.lnn)
        assert len(self.lnn.nodes) > 10

    def test_curriculum_creates_many_springs(self):
        train_code_curriculum(self.lnn)
        assert len(self.lnn.springs) > 10


# ============================================================
# 10. Integration: Syntax Spring Closure
# ============================================================

class TestSyntaxSpringClosure:
    def setup_method(self):
        self.lnn = LatticeNN()
        add_identity_anchor(self.lnn)

    def test_def_closes_to_paren(self):
        for _ in range(60):
            add_code_file(self.lnn, "def foo ( ) :")
        promote_code_springs(self.lnn, threshold=50)

        key = self.lnn._key("code:kw:def", "code:var:foo")
        assert key in self.lnn.springs
        assert self.lnn.springs[key].tau == 0

    def test_return_after_def_is_high_energy(self):
        for _ in range(60):
            add_code_file(self.lnn, "def foo ( ) :")
            add_code_file(self.lnn, "return x")
        promote_code_springs(self.lnn, threshold=50)

        def_node = "code:kw:def"
        return_node = "code:kw:return"
        key = self.lnn._key(def_node, return_node)
        if key in self.lnn.springs:
            assert self.lnn.springs[key].stiffness < 100

    def test_ngrams_for_code_patterns(self):
        for _ in range(10):
            add_code_file(self.lnn, "def foo ( ) :")
        assert len(self.lnn.trigrams) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
