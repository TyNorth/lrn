"""
College Corpus - advanced academic sentences
Covers: linear algebra, organic chemistry, quantum physics, computer science, economics, research methods, literary theory, cognitive science
"""

ORIGINAL_CORPUS = [
    # Linear Algebra
    "a vector space is a set of objects closed under addition and scalar multiplication",
    "linear independence means no vector in a set can be written as a linear combination of the others",
    "a basis is a linearly independent set of vectors that spans a vector space",
    "the dimension of a vector space is the number of vectors in any basis",
    "a linear transformation maps vectors from one vector space to another while preserving operations",
    "matrix multiplication corresponds to the composition of linear transformations",
    "the determinant of a square matrix indicates whether the matrix is invertible",
    "eigenvalues and eigenvectors characterize how a linear transformation scales certain directions",
    "the rank of a matrix equals the dimension of its column space",

    # Organic Chemistry
    "organic chemistry is the study of carbon containing compounds and their reactions",
    "hybridization describes the mixing of atomic orbitals to form new bonding orbitals",
    "stereoisomers have the same connectivity but differ in the spatial arrangement of atoms",
    "enantiomers are non superimposable mirror images of each other",
    "sn2 reactions proceed through a concerted mechanism with inversion of configuration",
    "aromatic compounds possess unusual stability due to delocalized pi electrons",
    "carbonyl compounds undergo nucleophilic addition and nucleophilic acyl substitution reactions",
    "spectroscopy techniques including nmr ir and mass spectrometry identify organic structures",

    # Quantum Physics
    "quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales",
    "wave particle duality states that particles exhibit both wave and particle properties",
    "the heisenberg uncertainty principle limits the precision with which position and momentum can be simultaneously known",
    "the schrodinger equation governs the time evolution of quantum systems",
    "the pauli exclusion principle states that no two fermions can occupy the same quantum state",
    "quantum tunneling allows particles to pass through energy barriers that would be classically forbidden",
    "superposition is the principle that a quantum system can exist in multiple states simultaneously",
    "entanglement describes correlations between particles that persist regardless of distance",

    # Computer Science
    "algorithms are step by step procedures for solving computational problems",
    "big o notation describes the upper bound on the time or space complexity of an algorithm",
    "binary search trees enable efficient search insertion and deletion in logarithmic time",
    "hash tables provide average case constant time lookup using key value pairs",
    "dynamic programming solves complex problems by breaking them into overlapping subproblems",
    "graph algorithms include breadth first search depth first search and dijkstras shortest path",
    "recursion solves problems by having a function call itself with a simpler input",
    "machine learning algorithms enable computers to learn patterns from data without explicit programming",

    # Economics
    "microeconomics studies the behavior of individual consumers firms and markets",
    "macroeconomics examines economy wide phenomena including inflation unemployment and growth",
    "marginal analysis compares the additional benefits and costs of a decision",
    "market failure occurs when free markets fail to allocate resources efficiently",
    "externalities are costs or benefits imposed on third parties not involved in a transaction",
    "game theory analyzes strategic interactions among rational decision makers",
    "the nash equilibrium is a set of strategies where no player benefits from unilaterally changing",
    "monetary policy uses interest rates and money supply to influence economic activity",

    # Research Methods
    "the scientific method involves observation hypothesis testing and theory development",
    "internal validity refers to the extent to which a study establishes a causal relationship",
    "external validity concerns the generalizability of findings to other settings and populations",
    "random assignment ensures that experimental and control groups are equivalent at the outset",
    "correlational studies examine relationships between variables without manipulating them",
    "longitudinal studies track the same subjects over an extended period of time",
    "qualitative research explores phenomena through interviews observations and textual analysis",
    "peer review evaluates research quality before publication in academic journals",

    # Literary Theory
    "formalism analyzes literary texts by focusing on their formal features rather than external context",
    "structuralism seeks to identify the underlying structures that organize cultural phenomena",
    "deconstruction reveals the internal contradictions and instabilities within texts",
    "feminist criticism examines how literature represents gender and reinforces or challenges patriarchy",
    "marxist criticism analyzes literature in terms of class relations and economic ideology",
    "postcolonial theory explores the cultural legacy of colonialism and imperialism in literature",
    "reader response theory emphasizes the role of the reader in creating meaning from a text",
    "intertextuality refers to the relationships and references between different texts",

    # Cognitive Science
    "cognitive science is the interdisciplinary study of mind and intelligence",
    "perception involves the interpretation of sensory information to construct a representation of the world",
    "working memory holds and manipulates information temporarily for cognitive tasks",
    "long term memory stores information over extended periods through encoding and retrieval processes",
    "problem solving requires representing the problem searching for solutions and evaluating outcomes",
    "heuristics are mental shortcuts that simplify decision making but can lead to systematic biases",
    "neural networks are computational models inspired by the structure of biological brains",
    "theory of mind is the ability to attribute mental states to oneself and others",
]

VARIED_EXAMPLES = [
    "find the eigenvalues of the matrix with entries one two three four",
    "the nullity plus rank of a matrix equals the number of columns",
    "diels alder reactions form six membered rings through cycloaddition",
    "retrosynthetic analysis works backward from the target molecule to available starting materials",
    "the double slit experiment demonstrates the wave nature of electrons",
    "bell inequalities test whether quantum mechanics violates local realism",
    "a balanced binary search tree has height logarithmic in the number of nodes",
    "p versus np asks whether every problem whose solution can be verified quickly can also be solved quickly",
    "the gini coefficient measures income inequality on a scale from zero to one",
    "behavioral economics incorporates psychological insights into economic models",
    "factor analysis reduces many variables into a smaller set of underlying factors",
    "the hermeneutic circle describes the interdependence of part and whole in interpretation",
    "the framing effect shows that how options are presented influences choices",
    "orthonormal vectors are perpendicular to each other and have unit length",
    "aldol condensation forms carbon carbon bonds between carbonyl compounds",
    "quantum computing uses qubits that can exist in superposition of zero and one",
    "red black trees maintain balance through coloring and rotation operations",
    "creative destruction describes how innovation replaces outdated economic structures",
    "thematic analysis identifies recurring patterns and themes in qualitative data",
    "the chinese room argument challenges the claim that computers truly understand language",
]

FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES
