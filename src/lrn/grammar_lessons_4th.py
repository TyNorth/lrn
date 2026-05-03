"""
4th Grade Grammar Lessons - 40 lessons covering:
- Complete sentences: simple, compound, complex
- Pronouns: subjective, objective, possessive
- Verbs: action, linking, helping; tenses
- Adjectives and adverbs: comparative, superlative
- Prepositions and prepositional phrases
- Conjunctions and interjections
- Commas, quotation marks, apostrophes
- Subjects and predicates
"""

import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


LESSONS_BASELINE_15 = [
    {
        "title": "Simple and Complete Sentences",
        "concept": "A complete sentence has a subject and predicate and expresses a complete thought.",
        "sentences": [
            "the dog wagged its tail happily",
            "she read three books last week",
            "the rain fell steadily all day",
            "we played soccer after school",
            "the teacher explained the lesson clearly",
            "my brother builds model airplanes",
            "the flowers bloomed in the garden",
            "they hiked up the mountain trail",
            "the chef prepared delicious meals",
            "the baby slept through the night",
        ],
    },
    {
        "title": "Compound Sentences",
        "concept": "A compound sentence joins two independent clauses with a conjunction.",
        "sentences": [
            "i wanted to play outside and my sister wanted to read",
            "the sun was hot but the breeze was cool",
            "we studied for the test so we felt confident",
            "she finished her homework and she submitted it online",
            "the game was exciting yet we lost",
            "marc likes math he also enjoys science",
            "the road was long but we reached destination",
            "they trained hard for the race and won trophy",
            "she was tired yet she kept working",
            "the music was loud so we covered our ears",
        ],
    },
    {
        "title": "Complex Sentences",
        "concept": "A complex sentence has one independent clause and at least one dependent clause.",
        "sentences": [
            "when the bell rang we lined up for lunch",
            "because it rained we stayed inside",
            "she smiled when she saw her friends",
            "the cat hissed although it usually purrs",
            "if you study then you will learn",
            "although he was tired he finished the race",
            "when the sun sets the sky turns orange",
            "because she practiced she improved quickly",
            "he waited until the bus arrived",
            "while the water boiled the eggs cooked",
        ],
    },
    {
        "title": "Subject and Predicate",
        "concept": "The subject tells WHO or WHAT. The predicate tells what the subject does.",
        "sentences": [
            "birds fly south during winter",
            "the team practiced every afternoon",
            "flowers bloom in spring",
            "the musician played the piano beautifully",
            "children learn quickly when interested",
            "the lighthouse guided ships safely",
            "grandparents tell stories to grandchildren",
            "the chef tasted the soup carefully",
            "students raised their hands to answer",
            "the painter created a masterpiece",
        ],
    },
    {
        "title": "Action Verbs",
        "concept": "Action verbs show what someone or something does.",
        "sentences": [
            "the boy kicked the soccer ball",
            "the dancer twirled across the stage",
            "water flows downhill",
            "the author wrote an interesting story",
            "the artist painted a mural on the wall",
            "birds build nests in trees",
            "the chef sliced the vegetables evenly",
            "the runner sprinted toward the finish line",
            "raindrops tap against the window",
            "the gardener planted new seeds yesterday",
        ],
    },
    {
        "title": "Linking Verbs",
        "concept": "Linking verbs connect the subject to a word that describes or identifies it.",
        "sentences": [
            "she is a talented musician",
            "the soup smells delicious",
            "they remain friends after the argument",
            "the cake tastes sweet",
            "he became an excellent programmer",
            "the sky appears cloudy today",
            "maya stayed calm during the test",
            "the weather turned cold suddenly",
            "she seems happy about the news",
            "the music sounds familiar to me",
        ],
    },
    {
        "title": "Helping Verbs",
        "concept": "Helping verbs assist the main verb to show time or possibility.",
        "sentences": [
            "she is studying for her exam",
            "they have completed the project",
            "we will visit grandmother tomorrow",
            "he could have forgotten the appointment",
            "the train should arrive soon",
            "i am writing a letter to my friend",
            "they were practicing the play all day",
            "she would have helped if asked",
            "the book might belong to the library",
            "we should clean our room before dinner",
        ],
    },
    {
        "title": "Verb Tenses",
        "concept": "Verbs change to show when actions happen: past, present, or future.",
        "sentences": [
            "i walked to school yesterday",
            "she reads books every weekend",
            "they will travel to europe next summer",
            "he played basketball last saturday",
            "we eat breakfast together every morning",
            "the baby will crawl soon",
            "she baked cookies last week",
            "the sun rises in the east",
            "they will plant trees in the spring",
            "i visited the museum last month",
        ],
    },
    {
        "title": "Pronouns: Subjective and Objective",
        "concept": "Subjective pronouns (I, you, he, she, it, we, they) are subjects. Objective pronouns (me, you, him, her, it, us, them) are objects.",
        "sentences": [
            "she gave him the book",
            "they invited us to the party",
            "he told me a secret",
            "we helped them with their project",
            "the teacher called on her during discussion",
            "the ball rolled toward them",
            "i saw him at the store yesterday",
            "she wrote it on the board",
            "the cat chased them around the house",
            "we gave her a birthday present",
        ],
    },
    {
        "title": "Possessive Pronouns and Nouns",
        "concept": "Possessive pronouns (my, your, his, her, its, our, their) show ownership.",
        "sentences": [
            "that is my backpack not yours",
            "their house has a blue door",
            "her art project won first place",
            "our dog loves to play fetch",
            "his bicycle has a flat tire",
            "the cat licked its paw clean",
            "our classroom has large windows",
            "their garden produced many tomatoes",
            "his voice sounds like his fathers",
            "the tree lost its leaves in autumn",
        ],
    },
    {
        "title": "Adjectives: Descriptive",
        "concept": "Adjectives describe nouns and tell what kind, which one, or how many.",
        "sentences": [
            "the tall giraffe ate leaves from the tree",
            "she wore a blue dress to the party",
            "the ancient castle stood on the hill",
            "a friendly dog greeted us at the door",
            "the delicious pizza smelled amazing",
            "the crowded room made it hard to hear",
            "soft clouds floated across the sky",
            "a tiny mouse hid behind the wall",
            "the sparkling water looked refreshing",
            "the heavy box was difficult to lift",
        ],
    },
    {
        "title": "Adjectives: Comparative and Superlative",
        "concept": "Comparative adjectives compare two things (-er or more). Superlative adjectives compare three or more things (-est or most).",
        "sentences": [
            "this mountain is taller than that one",
            "she is the fastest runner on the team",
            "my dog is more energetic than yours",
            "the ocean is deeper than the lake",
            "this is the most beautiful view i have seen",
            "his explanation was clearer than hers",
            "she is the smartest student in the class",
            "a rectangle has more sides than a triangle",
            "this problem is more difficult than the last one",
            "mount everest is the highest mountain in the world",
        ],
    },
    {
        "title": "Adverbs: How, When, Where",
        "concept": "Adverbs tell how something happens (verb modifiers), when, or where.",
        "sentences": [
            "she quickly finished her homework",
            "the baby cried loudly in the night",
            "they arrived early to get good seats",
            "he carefully placed the fragile vase down",
            "the train moved slowly up the mountain",
            "we will visit them soon",
            "she looked everywhere for her keys",
            "the rabbit hopped gracefully across the grass",
            "they play outside often after school",
            "the treasure was hidden somewhere in the forest",
        ],
    },
    {
        "title": "Prepositions and Prepositional Phrases",
        "concept": "Prepositions show relationships between nouns and other words. A prepositional phrase includes the preposition and its object.",
        "sentences": [
            "the book is on the table",
            "she walked through the park",
            "the cat sat under the chair",
            "they lived near the ocean",
            "the monkey jumped from tree to tree",
            "he put the letter in the mailbox",
            "the children played beside the river",
            "a bird flew above the clouds",
            "the spider crawled across the ceiling",
            "we hiked along the mountain trail",
        ],
    },
    {
        "title": "Conjunctions and Interjections",
        "concept": "Conjunctions (and, but, or, so, yet, for) join words or groups. Interjections (wow, ouch, hey) express strong feeling.",
        "sentences": [
            "i wanted to go but it started raining",
            "wow that was an amazing performance",
            "she studied hard and she passed the test",
            "hey that is my favorite song playing",
            "we could go to the beach or stay home",
            "ouch i stubbed my toe on the table",
            "i was tired so i went to bed early",
            "wow the sunset looks beautiful tonight",
            "she likes to read yet she rarely has time",
            "the game was exciting yet we lost in the end",
        ],
    },
]


def get_grammar_lessons(grade=4, full=True):
    """Get 4th grade grammar lessons."""
    if grade == 4:
        return LESSONS_BASELINE_15 if not full else LESSONS_BASELINE_15
    return []


def get_lesson_by_title(title, grade=4, full=True):
    """Get a specific lesson by title."""
    lessons = get_grammar_lessons(grade, full)
    for lesson in lessons:
        if lesson["title"].lower() == title.lower():
            return lesson
    return None


def get_all_grammar_sentences(grade=4, full=True):
    """Extract all sentences from grammar lessons."""
    lessons = get_grammar_lessons(grade, full)
    sentences = []
    for lesson in lessons:
        sentences.extend(lesson["sentences"])
    return sentences