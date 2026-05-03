"""
Sensory and Interoceptive Curriculum - K through 3rd Grade
Body awareness, self-regulation, and sensory processing concepts.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


LESSONS_K = [
    {
        "title": "My Body Has Feelings",
        "concept": "Our bodies send us signals about how we feel inside.",
        "sentences": [
            "my tummy feels rumbly when hungry",
            "my heart beats fast when excited",
            "my muscles feel tight when scared",
            "i feel warm when cozy",
            "my body tells me things",
            "my mouth feels dry when thirsty",
            "my eyes feel heavy when tired",
            "my skin feels prickly when cold",
            "i notice how my body feels",
            "my body sends me messages",
        ],
    },
    {
        "title": "Five Senses Fun",
        "concept": "We learn about the world through our five senses.",
        "sentences": [
            "i see colors bright",
            "i hear sounds loud",
            "i smell cookies sweet",
            "i taste apples juicy",
            "i feel textures soft",
            "my eyes see the sky blue",
            "my ears hear the birds sing",
            "my nose smells the flowers sweet",
            "my tongue tastes the lemon sour",
            "my hands feel the sand warm",
        ],
    },
    {
        "title": "Grounding With Our Senses",
        "concept": "When we feel mixed up, we can use our senses to feel better.",
        "sentences": [
            "i touch something soft",
            "i look at something pretty",
            "i listen to quiet sounds",
            "i smell something safe",
            "i taste something familiar",
            "i feel my feet on the ground",
            "i hear my voice",
            "i see five things around me",
            "i notice my breathing",
            "i feel my body right now",
        ],
    },
    {
        "title": "Fast and Slow Bodies",
        "concept": "Our bodies can feel fast like a race car or slow like a turtle.",
        "sentences": [
            "my heart races when i run",
            "my breathing slows when i rest",
            "i feel energetic like a bunny",
            "i feel calm like a sloth",
            "my body speeds up with excitement",
            "my body slows down with peace",
            "movement wakes up my body",
            "quiet calms my body",
            "fast heartbeat means i'm alive",
            "slow breathing means i'm safe",
        ],
    },
]

LESSONS_1ST = [
    {
        "title": "Inside Feelings",
        "concept": "Interoception helps us know what's happening inside our body.",
        "sentences": [
            "i feel butterflies in my tummy",
            "my chest feels tight when worried",
            "i sense my heartbeat in my body",
            "hunger feels like an empty feeling",
            "thirst feels like a dry feeling",
            "i notice my breath moving in and out",
            "my body temperature changes with activity",
            "i feel full after eating",
            "i recognize when i need to rest",
            "my body tells me when to stop",
        ],
    },
    {
        "title": "Sensory Toolkit",
        "concept": "We have tools to help us feel balanced when overwhelmed.",
        "sentences": [
            "deep pressure calms my body",
            "slow breathing settles my nerves",
            "looking at calm colors helps",
            "listening to soft music soothes",
            "holding something textured grounds me",
            "counting backwards calms my brain",
            "slow clapping synchronizes my brain",
            "fidget tools help me focus",
            "movement breaks up stuck feelings",
            "quiet space resets my system",
        ],
    },
    {
        "title": "Body Signals",
        "concept": "Learning to read our body's warning signs helps us stay safe.",
        "sentences": [
            "tummy tightening means worry",
            "face getting hot means embarrassment",
            "hands shaking means nervousness",
            "legs wanting to run means fear",
            "voice getting loud means excitement",
            "breathing getting fast means stress",
            "shoulders hunching means sadness",
            "jaw tightening means concentration",
            "body relaxing means safety",
            "muscles softening means calm",
        ],
    },
]

LESSONS_2ND = [
    {
        "title": "Interoception Skills",
        "concept": "Noticing and understanding our body's internal signals.",
        "sentences": [
            "i sense when i need to use the bathroom",
            "i notice when i'm hungry or full",
            "i recognize thirst before it becomes urgent",
            "i feel the difference between tired and bored",
            "i notice when my muscles need movement",
            "i sense when my body needs rest",
            "i recognize early signs of getting sick",
            "i notice temperature changes in my body",
            "i feel the difference between hunger and appetite",
            "i sense my heartbeat and know what it means",
        ],
    },
    {
        "title": "Sensory Processing",
        "concept": "How our brain organizes sensory information from our body and environment.",
        "sentences": [
            "some sounds feel too loud",
            "some textures feel uncomfortable",
            "bright lights feel overwhelming",
            "strong smells feel distracting",
            "movement helps my body feel alert",
            "quiet helps my body feel calm",
            "heavy work calms my nervous system",
            "fast movements excite my body",
            "slow movements settle my body",
            "balanced input helps me focus",
        ],
    },
    {
        "title": "Grounding Techniques",
        "concept": "Using our senses to come back to the present moment.",
        "sentences": [
            "i name five things i see",
            "i notice four things i can touch",
            "i listen for three sounds",
            "i smell two things around me",
            "i taste one thing in my mouth",
            "i feel my weight in my seat",
            "i notice the floor under my feet",
            "i feel the air on my skin",
            "i hear my own breathing",
            "i see where my hands are right now",
        ],
    },
]

LESSONS_3RD = [
    {
        "title": "Advanced Interoception",
        "concept": "Understanding complex body signals and emotions.",
        "sentences": [
            "i recognize the feeling before i get angry",
            "i notice anxiety in my chest before tests",
            "i sense excitement as energy in my body",
            "i feel disappointment as a heavy feeling",
            "i notice joy as warmth spreading",
            "i recognize frustration as tension building",
            "i sense calm as my body relaxing",
            "i notice nervousness as butterflies moving",
            "i feel proud as my chest opening up",
            "i recognize sadness as weight in my body",
        ],
    },
    {
        "title": "Self-Regulation Strategies",
        "concept": "Using body awareness to manage emotions and behaviors.",
        "sentences": [
            "when i notice tension i breathe deeply",
            "when i sense overwhelm i use sensory tools",
            "when i feel activation i move my body",
            "when i sense high energy i do heavy work",
            "when i feel low energy i change my posture",
            "when i notice scattered focus i ground myself",
            "when i sense rising stress i pause and notice",
            "when i feel stuck emotions i use movement",
            "when i notice anxiety i do counting exercises",
            "when i sense calm i can work on hard things",
        ],
    },
    {
        "title": "Body-Mind Connection",
        "concept": "Understanding how thoughts and feelings affect our body.",
        "sentences": [
            "worrying makes my stomach tight",
            "happy thoughts make my body feel light",
            "sad thoughts make my body feel heavy",
            "excited thoughts give me energy",
            "calm thoughts slow my breathing",
            "angry thoughts tense my muscles",
            "nervous thoughts speed up my heart",
            "peaceful thoughts relax my face",
            "confident thoughts straighten my posture",
            "relaxed thoughts soften my shoulders",
        ],
    },
]


def get_sensory_lessons(grade, full=False):
    """Get sensory/interoceptive lessons for a grade."""
    if grade == 0:
        return LESSONS_K[:4]
    elif grade == 1:
        return LESSONS_1ST[:4]
    elif grade == 2:
        return LESSONS_2ND[:4]
    elif grade == 3:
        return LESSONS_3RD[:4]
    return []


def get_all_sensory_concepts():
    """Get all sensory/interoceptive concepts."""
    concepts = []
    for grade in [0, 1, 2, 3]:
        for lesson in get_sensory_lessons(grade):
            concepts.extend(lesson.get("sentences", []))
    return concepts