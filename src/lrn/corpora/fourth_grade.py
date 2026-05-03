"""
4th Grade Corpus - Common Core aligned sentences
Covers: multi-digit arithmetic, fractions, decimals, geometry, reading comprehension,
writing, science (energy, waves, earth systems), social studies (colonial America, civics)
"""

ORIGINAL_CORPUS = [
    # Math - Multi-digit arithmetic
    "multiply two digit numbers using the standard algorithm",
    "divide four digit numbers by one digit divisors",
    "find the product of twenty three and fourteen",
    "solve division problems with remainders",
    "use place value to round multi digit numbers",
    "add and subtract multi digit whole numbers",
    "multiply a whole number up to four digits by a one digit number",
    "find whole number quotients with four digit dividends",
    "explain why multiplication algorithms work using place value",
    "use equations and rectangular arrays to model multiplication",

    # Math - Fractions
    "explain why a fraction a times n over b times n equals a over b",
    "compare two fractions with different numerators and denominators",
    "add and subtract fractions with like denominators",
    "multiply a fraction by a whole number",
    "understand decimal notation for fractions",
    "compare decimals to hundredths by reasoning about their size",
    "convert fractions with denominator ten to denominator hundred",
    "use visual models to understand equivalent fractions",
    "decompose a fraction into a sum of fractions with the same denominator",
    "solve word problems involving addition and subtraction of fractions",

    # Math - Measurement and Data
    "know relative sizes of measurement units within one system",
    "convert measurements from larger units to smaller units",
    "use the four operations to solve word problems involving distance and time",
    "apply the area and perimeter formulas for rectangles",
    "make a line plot to display a data set of measurements in fractions",
    "measure angles in whole number degrees using a protractor",
    "recognize angles as geometric shapes formed wherever two rays share an endpoint",
    "solve addition and subtraction problems to find unknown angles on a diagram",
    "understand that an angle that turns through n one degree angles has n degrees",

    # Math - Geometry
    "draw and identify points lines line segments rays angles",
    "identify parallel and perpendicular lines in two dimensional figures",
    "classify two dimensional figures based on the presence of parallel lines",
    "recognize a line of symmetry for a two dimensional figure",
    "identify line symmetric figures and draw lines of symmetry",
    "understand that attributes of a category of shapes also apply to subcategories",

    # Reading - Literature
    "refer to details and examples in a text when explaining what the text says explicitly",
    "determine the theme of a story from details in the text",
    "summarize a text using key details",
    "describe in depth a character setting or event in a story",
    "compare and contrast the point of view across multiple stories",
    "analyze how a series of chapters fits together to provide structure",
    "compare and contrast a firsthand and secondhand account of the same event",
    "read grade level prose and poetry accurately and with expression",

    # Reading - Informational Text
    "refer to details and examples in a text when explaining key ideas",
    "determine the main idea of a text and explain how it is supported by key details",
    "determine the meaning of general academic and domain specific words",
    "describe the overall structure of a text including chronology and comparison",
    "draw on information from multiple sources to answer a question efficiently",
    "explain how an author uses reasons and evidence to support particular points",
    "compare and contrast a firsthand and secondhand description of the same topic",
    "integrate information from two texts on the same topic to write knowledgeably",

    # Writing
    "write opinion pieces on topics with reasons and information",
    "write informative texts to examine a topic and convey ideas clearly",
    "write narratives to develop real or imagined experiences using effective technique",
    "produce clear and coherent writing appropriate to task purpose and audience",
    "with guidance and support from adults produce technology based writing",
    "conduct short research projects that build knowledge through investigation",
    "draw evidence from literary or informational texts to support analysis",
    "apply grade appropriate conventions of standard english capitalization and punctuation",

    # Science - Energy
    "energy can be moved from place to place by moving objects",
    "energy can be moved through sound light or electrical currents",
    "energy can be moved through heat thermal energy",
    "energy is present whenever there are moving objects or sound",
    "light can transfer energy and can be absorbed or reflected",
    "electricity and circuits can produce light heat and sound energy",
    "the faster an object moves the more energy it has",
    "collisions can cause energy to be transferred between objects",

    # Science - Waves
    "waves have properties of amplitude and wavelength",
    "waves can cause objects to move back and forth",
    "light travels in straight lines and can be reflected",
    "sound waves travel through different materials at different speeds",
    "digital information can be encoded using patterns of ones and zeros",
    "technology uses patterns to store and transmit information",

    # Science - Earth Systems
    "rainfall helps to shape the land and affects the types of living things found there",
    "water flows downhill and collects in oceans lakes and underground",
    "maps show the shapes of landforms and bodies of water in an area",
    "the distribution of water on earth is not uniform",
    "weather and climate are influenced by the sun and the oceans",
    "natural processes form canyons valleys and deltas over time",

    # Social Studies - Colonial America
    "the thirteen colonies were located along the atlantic coast",
    "colonists came to america for religious freedom and economic opportunity",
    "the mayflower compact was an early form of self government",
    "colonial regions developed different economies based on geography",
    "interactions between colonists and american indians were complex",
    "town meetings in new england were an early form of democracy",
    "the french and indian war changed the relationship between colonies and britain",
    "colonists began to resist british taxes and laws",

    # Social Studies - Civics
    "the constitution establishes the framework for american government",
    "the bill of rights protects individual freedoms",
    "citizens have rights and responsibilities in a democracy",
    "voting is a key responsibility of citizens in a democracy",
    "the three branches of government have different roles",
    "state and local governments make decisions that affect daily life",
    "rules and laws help maintain order and protect citizens",
    "civic participation strengthens democratic government",
]

VARIED_EXAMPLES = [
    "multiply thirty four by twenty two using the algorithm",
    "divide three thousand two hundred by eight",
    "find the product of fifty six and thirteen",
    "solve the division problem with a remainder of three",
    "round the number four thousand five hundred sixty seven to the nearest hundred",
    "add two thousand three hundred forty five and one thousand eight hundred ninety two",
    "compare the fractions three fourths and five eighths",
    "add two fifths plus one fifth",
    "multiply three times two sevenths",
    "write zero point seven five as a fraction",
    "compare zero point four five and zero point five four",
    "convert three tenths to thirty hundredths",
    "the area of a rectangle is length times width",
    "the perimeter of a rectangle is two times length plus two times width",
    "measure the angle using a protractor and record the degrees",
    "identify the parallel lines in the quadrilateral",
    "draw the line of symmetry for the butterfly shape",
    "energy moves when objects collide with each other",
    "sound waves travel through air water and solids",
    "light reflects off mirrors and shiny surfaces",
    "rainfall shapes rivers and carves canyons over time",
    "water flows from mountains down to the ocean",
    "the new england colonies had rocky soil and short growing seasons",
    "the southern colonies grew tobacco rice and indigo",
    "the middle colonies were known as the breadbasket colonies",
    "the first amendment protects freedom of speech and religion",
    "citizens serve on juries as part of their civic duty",
    "the legislative branch makes the laws",
    "the executive branch carries out the laws",
    "the judicial branch interprets the laws",
]

FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES
