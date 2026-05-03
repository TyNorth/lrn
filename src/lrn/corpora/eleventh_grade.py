"""
11th Grade Corpus - aligned sentences
Covers: algebra II, physics, US government, english 11, psychology
"""

ORIGINAL_CORPUS = [
    # Math - Algebra II
    "understand that the sum or product of two polynomials is also a polynomial",
    "apply the remainder and factor theorems to analyze polynomial functions",
    "identify zeros of polynomials and use them to construct rough graphs of the function",
    "rewrite rational expressions in different forms to solve problems",
    "solve rational equations and identify extraneous solutions",
    "understand the inverse relationship between exponents and logarithms",
    "solve exponential and logarithmic equations using properties of exponents and logarithms",
    "perform operations on complex numbers and represent them on the complex plane",
    "analyze the key features of trigonometric functions including period amplitude and midline",
    "graph trigonometric functions and their transformations",
    "prove and use trigonometric identities including the pythagorean identity",
    "understand sequences as functions with the domain being a subset of the integers",
    "use the binomial theorem to expand binomials raised to a positive integer power",

    # Science - Physics
    "newtons first law states that an object at rest stays at rest unless acted upon by a net force",
    "newtons second law relates force mass and acceleration through the equation f equals ma",
    "newtons third law states that for every action there is an equal and opposite reaction",
    "kinematics describes motion in terms of displacement velocity and acceleration",
    "projectile motion can be analyzed by separating horizontal and vertical components",
    "kinetic energy is the energy of motion and depends on mass and velocity",
    "the law of conservation of energy states that energy cannot be created or destroyed",
    "momentum is the product of mass and velocity and is conserved in isolated systems",
    "ohms law relates voltage current and resistance in an electrical circuit",
    "electromagnetic waves include radio waves visible light and x rays",
    "waves exhibit properties of reflection refraction diffraction and interference",

    # Social Studies - US Government & Politics
    "the constitution establishes a federal system with divided powers",
    "article one of the constitution establishes the legislative branch of government",
    "article two establishes the executive branch and the office of the president",
    "article three establishes the judicial branch and the supreme court",
    "judicial review allows courts to determine the constitutionality of laws and actions",
    "the first amendment protects freedom of religion speech press assembly and petition",
    "the fourteenth amendment guarantees equal protection and due process of law",
    "political parties organize candidates mobilize voters and develop policy platforms",
    "interest groups attempt to influence public policy through lobbying and advocacy",
    "civil liberties are individual freedoms protected from government interference",
    "the electoral college determines the outcome of presidential elections",

    # English 11 - American Literature & Composition
    "analyze how an authors choices about setting and structure contribute to meaning",
    "examine the development of american literary movements including transcendentalism and realism",
    "analyze the rhetorical strategies authors use to persuade and influence audiences",
    "evaluate the effectiveness of ethos pathos and logos in argumentative texts",
    "analyze how authors use diction syntax and imagery to create tone and mood",
    "examine how historical events shape the themes and perspectives of american literature",
    "write well organized arguments with precise claims and thorough analysis of evidence",
    "synthesize information from primary and secondary sources to support a thesis",
    "examine the american dream as a recurring theme in literature",
    "write analytical essays that interpret literary texts using textual evidence",

    # Psychology
    "psychology is the scientific study of behavior and mental processes",
    "the biological perspective examines how brain structures and neurotransmitters influence behavior",
    "classical conditioning pairs a neutral stimulus with an unconditioned stimulus to produce a response",
    "operant conditioning uses reinforcement and punishment to shape behavior",
    "the cognitive perspective studies how people think remember and solve problems",
    "memory involves encoding storing and retrieving information",
    "the psychodynamic perspective emphasizes unconscious drives and early childhood experiences",
    "the humanistic perspective focuses on personal growth self actualization and free will",
    "social psychology examines how individuals are influenced by the presence of others",
    "conformity is the tendency to adjust behavior to match group norms",
    "cognitive dissonance occurs when a person holds conflicting beliefs or attitudes",
    "therapeutic approaches include cognitive behavioral therapy psychoanalysis and humanistic therapy",
]

VARIED_EXAMPLES = [
    "factor the polynomial x cubed minus eight using the difference of cubes",
    "solve the logarithmic equation log base two of x equals five",
    "the amplitude of y equals three sin x is three",
    "expand the binomial x plus two to the fourth power",
    "a projectile launched at an angle follows a parabolic path",
    "the total mechanical energy is the sum of kinetic and potential energy",
    "calculate the force needed to accelerate a ten kilogram object at five meters per second squared",
    "voltage equals current times resistance according to ohms law",
    "marbury versus madison established the principle of judicial review",
    "the senate confirms presidential appointments and ratifies treaties",
    "analyze the use of irony in the short story",
    "transcendentalists like emerson and thoreau emphasized self reliance and nature",
    "the prefrontal cortex is responsible for decision making and impulse control",
    "positive reinforcement increases the likelihood of a behavior recurring",
    "the stanford prison experiment demonstrated the power of situational influences",
    "solve the equation two to the x equals thirty two",
    "the period of y equals sin two x is pi",
    "the separation of powers prevents concentration of authority",
    "the great gatsby critiques the corruption of the american dream",
    "nature versus nurture debates examine the relative influence of genetics and environment",
]

FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES
