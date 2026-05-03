"""
Computer Logic and Programming Foundations - K through 3rd Grade
Introduces computational thinking, sequencing, and basic programming concepts.

Grade K-1: Sequential thinking, patterns, step-by-step instructions
Grade 2-3: Basic logic (if/then), loops, variables, algorithmic thinking
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


LESSONS_K_1ST = [
    {
        "title": "Step by Step",
        "concept": "Instructions happen in order, one step at a time.",
        "sentences": [
            "first i wake up then i get dressed",
            "first i sit down then i open my book",
            "first i pick up the crayon then i draw a line",
            "first i walk to school then i see my friends",
            "first i eat breakfast then i brush my teeth",
            "step one is to put on shoes step two is to tie them",
            "first wash hands then eat lunch",
            "first find a pencil then write your name",
            "first open the door then walk through it",
            "first listen then speak",
        ],
    },
    {
        "title": "Patterns Everywhere",
        "concept": "Computers find and follow patterns.",
        "sentences": [
            "red blue red blue red blue goes the pattern",
            "circle square circle square circle square",
            "a b a b a b goes the letter pattern",
            "big bigger biggest follows a pattern",
            "one two three one two three counts in pattern",
            "morning noon evening morning noon evening is time pattern",
            "jump sit jump sit jump sit is movement pattern",
            "stop go stop go stop go is traffic pattern",
            "up down up down up down is motion pattern",
            "sunrise sunset sunrise sunset is day pattern",
        ],
    },
    {
        "title": "Find the Error",
        "concept": "Sometimes things go wrong and we need to find the mistake.",
        "sentences": [
            "i tried to pour juice but the cup was empty",
            "the picture has a line out of place",
            "the train went off the track at the turn",
            "i forgot to tie my shoe so it came off",
            "the sentence has a word in wrong order",
            "the blocks toppled because one was on wrong",
            "the pattern breaks because one shape is different",
            "the button did not work because it was not pressed",
            "the water spilled because the glass was too full",
            "the book fell because it was not on the shelf",
        ],
    },
    {
        "title": "Follow the Directions",
        "concept": "Clear instructions help us complete tasks.",
        "sentences": [
            "put the ball on the table not under it",
            "draw a circle then color it blue",
            "touch your head then touch your toes",
            "walk three steps forward then stop",
            "say your name then say your age",
            "stand up first then sit down",
            "write your letter first then draw a picture",
            "find a red block then put it in the box",
            "count to five then clap your hands",
            "look at the board then write what you see",
        ],
    },
]

LESSONS_2ND = [
    {
        "title": "If This Then That",
        "concept": "Basic conditional logic: when something happens, something else follows.",
        "sentences": [
            "if it rains then i need an umbrella",
            "if i am hungry then i eat food",
            "if the light is red then i stop",
            "if the light is green then i go",
            "if i am tired then i rest",
            "if the door is closed then i knock first",
            "if my pencil breaks then i sharpen it",
            "if i finish my work then i read a book",
            "if it is cold outside then i wear a coat",
            "if i make a mistake then i erase and try again",
        ],
    },
    {
        "title": "Repeat the Pattern",
        "concept": "Loops repeat actions over and over.",
        "sentences": [
            "count from one to ten again and again",
            " clap clap clap clap clap again and again",
            "walk to the door and back again and again",
            "stir the pot stir the pot stir the pot",
            "jump up and down jump up and down jump up and down",
            "read the word read the word read the word",
            "check the clock check the clock check the clock",
            "water the plant water the plant water the plant",
            "practice the piano practice the piano practice the piano",
            "line up line up line up line up line up",
        ],
    },
    {
        "title": "Put It Together",
        "concept": "Complex tasks break into simpler steps.",
        "sentences": [
            "to make a sandwich first get bread then add filling then close it",
            "to clean a room pick up toys put them away wipe surfaces sweep floor",
            "to get dressed take off pajamas put on shirt put on pants put on shoes",
            "to brush teeth get toothbrush apply paste brush teeth rinse mouth",
            "to write a story think of idea write beginning write middle write end",
            "to solve puzzle look at pieces find corners find edges find middle",
            "to plant a seed dig hole put seed cover hole water seed",
            "to build tower pick block place block pick block place block",
            "to draw a house draw square add triangle draw door draw windows",
            "to make art gather supplies draw outline add color add details",
        ],
    },
    {
        "title": "True or False",
        "concept": "Statements can be correct or incorrect.",
        "sentences": [
            "the sky is blue true or false true",
            "cats can fly false",
            "two plus two equals four true",
            "the ocean is wet true",
            "the sun is made of cheese false",
            "monday comes after sunday false",
            "apples grow on trees true",
            "fish can swim true",
            "rocks can dance false",
            "reading is fun true",
        ],
    },
]

LESSONS_3RD = [
    {
        "title": "Conditions and Choices",
        "concept": "Programs make decisions based on conditions.",
        "sentences": [
            "if temperature is above one hundred then water boils",
            "if score is greater than ninety then grade is an a",
            "if age is eighteen or older then can vote",
            "if light is dim then eyes strain",
            "if answer is wrong then try again",
            "if box is empty then fill it",
            "if button is pressed then something happens",
            "if password is correct then access is granted",
            "if time is up then game ends",
            "if path is blocked then find another way",
        ],
    },
    {
        "title": "Variables Store Things",
        "concept": "Variables hold information that can change.",
        "sentences": [
            "my age changes every year",
            "the score changes every time points are earned",
            "the weather changes from day to day",
            "my height changes as i grow",
            "the count goes up when we add more",
            "the temperature changes with the season",
            "the answer changes when numbers change",
            "the location changes when we move",
            "the name stays the same but address might change",
            "the total changes when we add or subtract",
        ],
    },
    {
        "title": "Loops and Repetition",
        "concept": "Loops let us repeat actions without retyping.",
        "sentences": [
            "repeat the chorus three times",
            "keep stirring until soup is hot",
            "keep practicing until perfect",
            "loop around the track five laps",
            "repeat the refrain in the song",
            "keep counting until reaching one hundred",
            "repeat the pattern for the whole row",
            "keep adding until reaching the total",
            "repeat the exercise ten times",
            "loop through the alphabet from a to z",
        ],
    },
    {
        "title": "Break It Down",
        "concept": "Decomposition: big problems become small problems.",
        "sentences": [
            "making a movie means writing script and casting actors and filming scenes and editing",
            "building a house means laying foundation and framing walls and installing roof and painting",
            "planning a party means choosing theme and sending invites and preparing food and decorating",
            "reading a book means looking at cover and reading pages and answering questions and discussing",
            "cooking meal means gathering ingredients and preparing food and cooking and serving",
            "solving math problem means reading question and finding information and calculating and checking",
            "writing report means researching topic and organizing ideas and writing draft and editing",
            "making bed means smoothing sheets and arranging pillows and straightening blanket",
            "cleaning car means removing trash and wiping surfaces and vacuuming inside and washing outside",
            "learning topic means reading material and taking notes and practicing skills and testing knowledge",
        ],
    },
]


def get_programming_lessons(grade, full=False):
    """Get computer logic/programming lessons for a grade."""
    if grade <= 1:
        return LESSONS_K_1ST[:4]
    elif grade == 2:
        return LESSONS_2ND[:4]
    elif grade == 3:
        return LESSONS_3RD[:4]
    return []


def get_all_programming_sentences(grade):
    """Get all programming-related sentences for a grade."""
    lessons = get_programming_lessons(grade)
    sentences = []
    for lesson in lessons:
        sentences.extend(lesson.get("sentences", []))
    return sentences