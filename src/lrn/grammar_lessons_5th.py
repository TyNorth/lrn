"""
5th Grade Grammar Lessons - Focus on writing skills:
- Verbals: participles, gerunds, infinitives
- clauses: independent, dependent, noun clauses
- pronoun-antecedent agreement
- precise word choice and vocabulary
- sentence combining and embedding
- punctuation: semicolons, colons, dashes
- quotation marks in dialogue
"""

import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


LESSONS_BASELINE_15 = [
    {
        "title": "Verbals: Participles",
        "concept": "Participles are verb forms used as adjectives. Present participles end in -ing. Past participles often end in -ed.",
        "sentences": [
            "the glowing candle lit up the room",
            "exhausted from running the boy collapsed on the bench",
            "the broken vase lay in pieces on the floor",
            "the rising sun cast long shadows across the field",
            "the singing bird perched on the branch",
            "the frightened child hid behind her mother",
            "the frozen lake could support the weight of the snow",
            "the waving flag rippled in the wind",
            "the completed project exceeded expectations",
            "the surprising news arrived late in the evening",
        ],
    },
    {
        "title": "Verbals: Gerunds",
        "concept": "Gerunds are verb forms ending in -ing that function as nouns.",
        "sentences": [
            "swimming is excellent exercise for the whole body",
            "the building of the pyramid took many years",
            "her singing impressed the entire audience",
            "running requires proper shoes and stretching",
            "the cooking class taught us new recipes",
            "his constant complaining grew tiresome",
            "the finding of the treasure made them wealthy",
            "reading improves vocabulary and comprehension",
            "the sculpting of the statue required great skill",
            "studying every day leads to better test scores",
        ],
    },
    {
        "title": "Verbals: Infinitives",
        "concept": "Infinitives are to + verb forms that function as nouns, adjectives, or adverbs.",
        "sentences": [
            "to err is human to forgive is divine",
            "she wanted to travel around the world",
            "the decision to leave was difficult to make",
            "he gave money to help the charity",
            "the plan to expand seemed promising",
            "to succeed requires dedication and hard work",
            "she had a book to read during the trip",
            "the ability to think critically is important",
            "he agreed to meet us at the cafe",
            "the chance to win motivated the team",
        ],
    },
    {
        "title": "Independent and Dependent Clauses",
        "concept": "Independent clauses can stand alone. Dependent clauses cannot stand alone and need an independent clause.",
        "sentences": [
            "because it rained we cancelled the picnic",
            "when the bell rings the students exit the classroom",
            "the dog barked although it usually sleeps during the day",
            "since she studied she passed the exam easily",
            "while the soup cooled the family set the table",
            "the team practiced because they wanted to win",
            "unless you ask you will not receive help",
            "after the sun set we gathered around the fire",
            "the girl cried when she lost her favorite toy",
            "they continued working although they felt exhausted",
        ],
    },
    {
        "title": "Noun Clauses",
        "concept": "Noun clauses function as nouns and can be subjects, objects, or complements.",
        "sentences": [
            "what he said surprised everyone in the room",
            "that she won the award made her extremely happy",
            "where the treasure was hidden remained a mystery",
            "whoever finishes first will receive a prize",
            "when the meeting starts depends on the speaker",
            "whether she comes or not does not change our plans",
            "whomever you choose will lead the team",
            "whatever you decide will be acceptable to us",
            "how the machine works is explained in the manual",
            "why the project failed needs careful analysis",
        ],
    },
    {
        "title": "Pronoun-Antecedent Agreement",
        "concept": "Pronouns must agree with their antecedents in number and gender.",
        "sentences": [
            "each student must bring his or her textbook to class",
            "the committee made its decision unanimously",
            "neither the boys nor the girl forgot her homework",
            "someone left their jacket on the chair",
            "every parent hopes their child succeeds",
            "the team celebrated its victory with enthusiasm",
            "either the cats or the dog has eaten its food",
            "each of the players gave their best effort",
            "the school announced its new policy yesterday",
            "no one remembered to bring their lunch today",
        ],
    },
    {
        "title": "Precise Word Choice",
        "concept": "Using precise words makes writing clearer and more vivid.",
        "sentences": [
            "the water glistened rather than just shone",
            "she whispered softly not just spoke quietly",
            "the athlete ran swiftly across the finish line",
            "the old house creaked quietly in the wind",
            "the chef prepared the meal carefully and precisely",
            "the storm raged violently through the town",
            "the child laughed happily at the funny clown",
            "the mountain stood majestically against the sky",
            "the detective examined the clue thoroughly",
            "the author wrote eloquently about the subject",
        ],
    },
    {
        "title": "Sentence Combining",
        "concept": "Combine short sentences using conjunctions, relative clauses, or appositives.",
        "sentences": [
            "the girl plays piano she also sings beautifully",
            "the professor taught the class the professor also mentored students",
            "marcus is talented he also shows great dedication",
            "the book was interesting it was also educational",
            "she arrived late she still managed to finish on time",
            "the movie had action it also had comedy",
            "the artist painted portraits she also sculpted statues",
            "the program teaches skills it also builds confidence",
            "the city grew rapidly it faced challenges too",
            "he trained hard he earned the championship finally",
        ],
    },
    {
        "title": "Semicolons and Colons",
        "concept": "Semicolons join related independent clauses. Colons introduce lists, explanations, or quotations.",
        "sentences": [
            "the recipe requires flour; it also needs sugar",
            "we visited many countries: france, spain, and italy",
            "she had one goal: to graduate with honors",
            "the rules are simple; follow them carefully",
            "bring supplies: pencils, paper, and erasers",
            "the原因是 clear: we need more time to finish",
            "he succeeded through hard work; talent alone was not enough",
            "the ingredients include: flour, water, and yeast",
            "there is one thing you must remember: be yourself",
            "the team practiced daily; they won the championship",
        ],
    },
    {
        "title": "Quotation Marks in Dialogue",
        "concept": "Use quotation marks to indicate a speaker's exact words. Capitalize the first word of dialogue.",
        "sentences": [
            "i love reading books, said the young girl excitedly",
            "will you come to my party, asked sam, on saturday",
            "the teacher said, work quietly during the test",
            "what time is it, asked the student looking at the clock",
            "good job, the coach shouted, you did it",
            "she whispered, i am scared of the dark",
            "the sign read, wet floor caution",
            "are you finished, asked the mom, with your homework",
            "the book says, knowledge is power",
            "he shouted, watch out for the car",
        ],
    },
    {
        "title": "Dashes and Hyphens",
        "concept": "Dashes create emphasis or set off parenthetical information. Hyphens join compound words.",
        "sentences": [
            "the winner the only one who finished early got the trophy",
            "she is a well-known author across the country",
            "the twenty-first century brought new challenges",
            "the problem-solver found a creative solution",
            "his explanation was clear-cut and easy to understand",
            "the long-awaited vacation finally arrived",
            "the child-like wonder never left her eyes",
            "three-fourths of the class passed the test",
            "the self-correcting system fixed its own errors",
            "the high-speed chase ended safely",
        ],
    },
    {
        "title": "Parallel Structure",
        "concept": "Use parallel structure with coordinating conjunctions to create balance.",
        "sentences": [
            "she likes hiking swimming and biking",
            "the book is interesting informative and well-written",
            "to succeed you need talent to succeed you need discipline",
            "the job requires creativity intelligence and experience",
            "he learned to read to study and to teach",
            "the program is affordable practical and convenient",
            "running jumping and swimming are great exercises",
            "the scientist observed recorded and analyzed the data",
            "she is smart talented and determined",
            "the course teaches theory practice and application",
        ],
    },
    {
        "title": "Active and Passive Voice",
        "concept": "Active voice: subject does the action. Passive voice: subject receives the action.",
        "sentences": [
            "the dog bit the man yesterday active voice",
            "the man was bitten by the dog yesterday passive voice",
            "she wrote the novel in two months active voice",
            "the novel was written by her in two months passive voice",
            "the teacher explained the lesson clearly active voice",
            "the lesson was explained clearly by the teacher passive voice",
            "the chef prepared an amazing meal active voice",
            "an amazing meal was prepared by the chef passive voice",
            "the storm destroyed the old barn active voice",
            "the old barn was destroyed by the storm passive voice",
        ],
    },
    {
        "title": "Sentence Fragments and Run-ons",
        "concept": "A complete sentence has a subject and predicate. Fragments lack one. Run-ons have too many clauses.",
        "sentences": [
            "because it was raining fragment needs independent clause",
            "the boy who lost his homework had to redo it complete",
            "i went to the store i bought milk run-on needs semicolon or period",
            "she studied for the test she passed with a high score complete sentence",
            "when the bell rang fragment dependent clause only",
            "the cat slept on the couch all day complete",
            "he likes to read he also likes to write complete but choppy",
            "after the sun set we made dinner complete sentence",
            "the book is interesting run-on needs more information",
            "while waiting for the bus i read a book complete",
        ],
    },
    {
        "title": "Capitalization Rules",
        "concept": "Capitalize proper nouns, proper adjectives, first word of sentences, and titles.",
        "sentences": [
            "the capital of france is paris and it is a beautiful city",
            "dr smith is a well-respected physician at the hospital",
            "the amazon river flows through the rain forest in south america",
            "my favorite holiday is christmas because i love the winter season",
            "president lincoln gave the gettysburg address in eighteen sixty-three",
            "mount everest is the highest mountain in the world located in the himalayas",
            "shakespeare wrote many famous plays including hamlet and romeo and juliet",
            "the united nations works to promote peace and cooperation among nations",
            "the golden gate bridge is an iconic landmark in san francisco california",
            "mrs johnson teaches english at lincoln high school on oak street",
        ],
    },
]


def get_grammar_lessons(grade=5, full=True):
    if grade == 5:
        return LESSONS_BASELINE_15 if not full else LESSONS_BASELINE_15
    return []


def get_lesson_by_title(title, grade=5, full=True):
    lessons = get_grammar_lessons(grade, full)
    for lesson in lessons:
        if lesson["title"].lower() == title.lower():
            return lesson
    return None


def get_all_grammar_sentences(grade=5, full=True):
    lessons = get_grammar_lessons(grade, full)
    sentences = []
    for lesson in lessons:
        sentences.extend(lesson["sentences"])
    return sentences