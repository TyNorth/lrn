"""
Pre-K English Corpus - Comprehensive Curriculum Coverage

Based on standard Pre-K frameworks (DIBELS, TPRI, Common Core Pre-K, Head Start ELOF)
Covers: Literacy, Mathematics, Science, Social Studies, Social-Emotional, Physical, Arts
"""

# ============================================================
# ORIGINAL CORPUS - Core curriculum sentences
# ============================================================

ORIGINAL_CORPUS = [
    # === LETTERS & PHONICS ===
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A is for apple", "B is for ball", "C is for cat",
    "D is for dog", "E is for egg", "F is for fish",
    "G is for goat", "H is for hat", "I is for ice",
    "J is for jam", "K is for kite", "L is for lion",
    "M is for moon", "N is for nest", "O is for orange",
    "P is for pig", "Q is for queen", "R is for rat",
    "S is for sun", "T is for tree", "U is for umbrella",
    "V is for van", "W is for watch", "X is for box",
    "Y is for yellow", "Z is for zebra",

    # Letter sounds
    "A says ah", "B says buh", "C says kuh",
    "D says duh", "E says eh", "F says fuh",
    "G says guh", "H says huh", "I says ih",
    "J says juh", "K says kuh", "L says luh",
    "M says muh", "N says nuh", "O says oh",
    "P says puh", "Q says kwuh", "R says ruh",
    "S says suh", "T says tuh", "U says uh",
    "V says vuh", "W says wuh", "X says ksuh",
    "Y says yuh", "Z says zuh",

    # === NUMBERS & COUNTING ===
    "zero one two three four five six seven eight nine ten",
    "eleven twelve thirteen fourteen fifteen",
    "sixteen seventeen eighteen nineteen twenty",
    "i can count to ten", "i can count to twenty",
    "one two three four five",
    "six seven eight nine ten",
    "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
    "i have one apple", "i have two cats", "i have three dogs",
    "i see four birds", "i see five fish", "i count six stars",
    "seven is lucky", "eight is great", "nine is fine", "ten is the end",
    "count the blocks one two three",
    "how many fingers do i have ten",
    "zero means none", "one means just one",

    # === COLORS ===
    "red blue green yellow orange purple",
    "black white brown pink gray gold silver",
    "the apple is red", "the sky is blue", "the grass is green",
    "the sun is yellow", "the orange is orange", "the grape is purple",
    "the night is black", "the snow is white", "the bear is brown",
    "the flower is pink", "the sky is gray", "the ring is gold",
    "i see something red", "i see something blue",
    "what color is this it is green",

    # === SHAPES ===
    "circle square triangle rectangle oval",
    "star heart diamond pentagon hexagon",
    "the ball is a circle", "the box is a square",
    "the roof is a triangle", "the door is a rectangle",
    "the egg is an oval", "the star is a star",
    "the heart is a heart", "the diamond is a diamond",
    "a circle is round", "a square has four sides",
    "a triangle has three sides",

    # === ANIMALS ===
    "cat dog bird fish horse cow",
    "sheep pig chicken duck rabbit",
    "lion tiger bear elephant monkey giraffe",
    "snake turtle frog butterfly bee",
    "the cat says meow", "the dog says woof",
    "the bird says tweet", "the cow says moo",
    "the pig says oink", "the duck says quack",
    "the horse says neigh", "the sheep says baa",
    "the lion says roar", "the frog says ribbit",

    # === EMOTIONS ===
    "happy sad angry scared surprised tired",
    "excited proud frustrated calm nervous",
    "i feel happy", "i feel sad", "i feel angry",
    "i feel scared", "i feel surprised", "i feel tired",
    "i feel excited", "i feel proud",
    "happy is good", "sad is okay",
    "when i am angry i take a breath",
    "when i am scared i ask for help",

    # === BODY PARTS ===
    "head shoulders knees and toes",
    "eyes ears mouth nose",
    "hands fingers arms legs feet",
    "tummy elbows chin hair cheeks",
    "i have two eyes", "i have two ears",
    "i have one nose", "i have one mouth",
    "i have two hands", "i have ten fingers",
    "i have two feet", "i have ten toes",
    "i use my eyes to see", "i use my ears to hear",
    "i use my nose to smell", "i use my mouth to taste",
    "i use my hands to touch",

    # === FIVE SENSES ===
    "i see with my eyes", "i hear with my ears",
    "i smell with my nose", "i taste with my tongue",
    "i touch with my hands",
    "i see the sun", "i hear the bird",
    "i smell the flower", "i taste the apple",
    "i touch the soft blanket",

    # === FAMILY ===
    "mom dad sister brother baby",
    "grandma grandpa aunt uncle cousin",
    "i love my family", "my family loves me",
    "my mom takes care of me", "my dad plays with me",
    "my sister is big", "my brother is little",
    "my grandma makes cookies", "my grandpa reads to me",

    # === COMMUNITY ===
    "teacher doctor firefighter police officer",
    "librarian mail carrier farmer baker",
    "the teacher helps me learn",
    "the doctor helps me stay healthy",
    "the firefighter puts out fires",
    "the police officer keeps us safe",
    "the librarian reads stories",
    "the farmer grows food",

    # === WEATHER & SEASONS ===
    "sunny cloudy rainy snowy windy stormy",
    "spring summer fall winter",
    "the sun is shining", "the clouds are gray",
    "the rain is falling", "the snow is cold",
    "the wind is blowing", "the storm is loud",
    "in spring flowers grow", "in summer it is hot",
    "in fall leaves fall down", "in winter it is cold",

    # === PLANTS & GROWING ===
    "seed root stem leaf flower tree",
    "the seed goes in the ground",
    "the root drinks water", "the stem grows up",
    "the leaf catches sun", "the flower blooms",
    "the tree is big", "plants need water and sun",

    # === DAYS & TIME ===
    "morning afternoon night",
    "today tomorrow yesterday",
    "monday tuesday wednesday thursday friday saturday sunday",
    "good morning", "good afternoon", "good night",
    "i wake up in the morning",
    "i eat lunch in the afternoon",
    "i sleep at night",
    "today is a good day", "tomorrow we will play",

    # === SIGHT WORDS & SIMPLE SENTENCES ===
    "i see the cat", "i see the dog",
    "the cat is big", "the dog is small",
    "i like the sun", "i like the moon",
    "the bird can fly", "the fish can swim",
    "i can run", "i can jump", "i can play",
    "this is my book", "that is your toy",
    "we are friends", "you are kind",
    "he is tall", "she is fast", "it is red",
    "they are playing",

    # === RHYME FAMILIES ===
    "cat hat mat sat bat fat",
    "dog log fog hog jog",
    "bed red fed led shed",
    "sun run fun bun pun",
    "big pig dig wig fig",
    "hot pot lot dot got",
    "can fan man pan van",
    "pen hen ten men den",
    "cap map lap nap tap",
    "sit fit hit kit bit",

    # === POSITION WORDS ===
    "the cat is on the mat",
    "the dog is under the table",
    "the bird is in the tree",
    "the ball is beside the box",
    "the toy is between the books",
    "the sun is above the clouds",
    "the fish is below the water",
    "the car is next to the house",
    "the cat is behind the chair",
    "the dog is in front of the door",

    # === COMPARISONS ===
    "big and small", "long and short",
    "tall and short", "heavy and light",
    "fast and slow", "hot and cold",
    "full and empty", "hard and soft",
    "loud and quiet", "bright and dark",
    "more and less", "many and few",
    "the elephant is big", "the mouse is small",
    "the giraffe is tall", "the cat is short",
    "the rock is heavy", "the feather is light",
    "the car is fast", "the turtle is slow",

    # === PATTERNS ===
    "red blue red blue red blue",
    "circle square circle square",
    "big small big small big small",
    "one two one two one two",
    "cat dog cat dog cat dog",
    "a b a b a b",

    # === SORTING & CLASSIFYING ===
    "sort by color", "sort by shape",
    "sort by size", "sort by type",
    "all the red things go together",
    "all the circles go together",
    "all the big things go together",
    "animals go here plants go there",

    # === HEALTH & SAFETY ===
    "wash your hands", "brush your teeth",
    "wear a helmet", "buckle your seatbelt",
    "look both ways", "stop and go",
    "hot do not touch", "sharp be careful",
    "say please", "say thank you",
    "share with friends", "take turns",
    "use your words", "ask for help",

    # === MOVEMENT & PHYSICAL ===
    "run jump hop skip",
    "climb throw catch kick",
    "balance crawl slide swing",
    "clap stomp dance march",
    "i can run fast", "i can jump high",
    "i can hop on one foot", "i can skip to school",

    # === MUSIC & ART ===
    "sing a song", "dance to music",
    "clap your hands", "stomp your feet",
    "fast music slow music",
    "loud music quiet music",
    "draw a picture", "paint a rainbow",
    "cut with scissors", "paste on paper",
    "color inside the lines", "make something new",

    # === SOCIAL SKILLS ===
    "my name is", "i am years old",
    "i can do it myself", "i try my best",
    "i can wait my turn", "i can share my toys",
    "i can help my friend", "i can say sorry",
    "i can use gentle hands", "i can use walking feet",
    "i can clean up", "i can line up",
    "i can raise my hand", "i can listen",

    # === LIVING VS NON-LIVING ===
    "living things grow", "living things need food",
    "living things breathe", "living things move",
    "a cat is living", "a tree is living",
    "a rock is not living", "a toy is not living",
    "plants are living", "animals are living",

    # === MONEY (basic) ===
    "penny nickel dime quarter",
    "a penny is one cent", "a nickel is five cents",
    "a dime is ten cents", "a quarter is twenty five cents",
    "money buys things", "we save money",

    # === PRETEND PLAY ===
    "let us pretend", "i am a doctor",
    "i am a teacher", "i am a firefighter",
    "i am a chef", "i am a pilot",
    "we play house", "we play store",
    "we play school", "we play restaurant",
]


# ============================================================
# VARIED EXAMPLES - Reinforcement and variation
# ============================================================

VARIED_EXAMPLES = [
    # Letter reinforcement
    "A a apple ant arm", "B b ball bat bed",
    "C c cat cup car", "D d dog door day",
    "E e egg eye ear", "F f fish fan fun",
    "G g goat game go", "H h hat hand hot",
    "I i ice igloo in", "J j jam jar job",
    "K k kite key kid", "L l lion leg log",
    "M m moon man mat", "N n nest net no",
    "O o orange on off", "P p pig pen pan",
    "Q q queen quick quiet", "R r rat run red",
    "S s sun sit sad", "T t tree top ten",
    "U u umbrella up us", "V v van van very",
    "W w watch web we", "X x box fox six",
    "Y y yellow yes you", "Z z zebra zoo zero",

    # Number reinforcement
    "zero zero zero", "one one one", "two two two",
    "three three three", "four four four", "five five five",
    "six six six", "seven seven seven", "eight eight eight",
    "nine nine nine", "ten ten ten",
    "one two three", "two three four",
    "three four five", "four five six",
    "five six seven", "six seven eight",
    "seven eight nine", "eight nine ten",
    "ten eleven twelve", "thirteen fourteen fifteen",
    "sixteen seventeen eighteen", "nineteen twenty",

    # Color reinforcement
    "red red red", "blue blue blue", "green green green",
    "yellow yellow yellow", "orange orange orange",
    "purple purple purple", "black black black",
    "white white white", "brown brown brown",
    "pink pink pink", "gray gray gray",

    # Shape reinforcement
    "circle circle circle", "square square square",
    "triangle triangle triangle", "rectangle rectangle rectangle",
    "oval oval oval", "star star star",
    "heart heart heart", "diamond diamond diamond",

    # Animal reinforcement
    "cat cat cat", "dog dog dog", "bird bird bird",
    "fish fish fish", "horse horse horse", "cow cow cow",
    "sheep sheep sheep", "pig pig pig",
    "chicken chicken chicken", "duck duck duck",

    # Emotion reinforcement
    "happy happy happy", "sad sad sad",
    "angry angry angry", "scared scared scared",
    "excited excited excited", "calm calm calm",

    # Body part reinforcement
    "head head head", "eyes eyes eyes",
    "nose nose nose", "mouth mouth mouth",
    "hands hands hands", "feet feet feet",

    # Rhyme family variations
    "the cat has a hat", "the cat sits on the mat",
    "the cat sat on the mat", "the bat hit the cat",
    "a fat cat sat on a mat", "the rat sat on a hat",
    "the dog ran over the log", "the fog is on the hog",
    "the dog can jog", "a big dog sat on a log",
    "the bed is red", "i fed the cat on the bed",
    "the red bed is big", "i led the dog to the bed",
    "the sun is hot", "i run in the sun",
    "the sun is fun", "i have a bun in the sun",
    "the big pig can dig", "the pig has a big wig",
    "a big pig sat on a fig", "the pig can dig big",
    "the pot is hot", "i got a hot pot",
    "the hot pot has a lot", "a lot of dots on the pot",
    "the man has a can", "the fan is on the man",
    "the man ran with a pan", "a van ran to the man",
    "the hen has a pen", "ten men have a pen",
    "the hen is in the den", "ten men ran to the hen",
    "i sit on the mat", "the cat can sit",
    "i hit the ball", "the kit is a fit",
    "the cap is on the map", "i nap on the lap",
    "tap the cap on the map", "the cat naps on the lap",

    # Sentence variations
    "i see the cat", "i see the dog", "i see the bird",
    "i see the fish", "i see the sun", "i see the moon",
    "i see a big cat", "i see a small dog",
    "i see the red ball", "i see the blue sky",
    "i see the green grass", "i see the yellow sun",
    "i like the cat", "i like the dog", "i like the bird",
    "i like to run", "i like to play", "i like to eat",
    "i like the big cat", "i like the small dog",
    "i like the red apple", "i like the blue sky",
    "the cat is big", "the dog is small", "the bird is fast",
    "the fish is small", "the sun is hot", "the moon is bright",
    "the cat is on the mat", "the dog is on the log",
    "the bird is in the tree", "the fish is in the water",
    "the sun is in the sky", "the moon is in the sky",
    "i can see the cat", "i can see the dog",
    "i can run fast", "i can jump high",
    "i can play with the cat", "i can play with the dog",
    "i can eat the apple", "i can drink the water",
    "the cat runs", "the dog runs", "the bird flies",
    "the fish swims", "the horse runs", "the cow eats",
    "the cat sits on the mat", "the dog sits on the log",
    "the bird sits in the tree", "the fish swims in the water",
    "where is the cat", "where is the dog",
    "what is that", "who is there",
    "can you see the cat", "can you see the dog",
    "do you like the cat", "do you like the dog",

    # Pattern variations
    "red blue green red blue green",
    "big small medium big small medium",
    "one two three one two three",
    "cat dog bird cat dog bird",
    "a b c a b c",

    # Position word variations
    "the book is on the table",
    "the cat is under the bed",
    "the bird is in the cage",
    "the ball is beside the door",
    "the toy is between the pillows",
    "the clock is above the door",
    "the rug is below the table",
    "the car is next to the tree",
    "the mouse is behind the wall",
    "the dog is in front of the house",

    # Comparison variations
    "the elephant is bigger than the cat",
    "the mouse is smaller than the dog",
    "the rope is longer than the string",
    "the pencil is shorter than the ruler",
    "the tree is taller than the bush",
    "the rock is heavier than the leaf",
    "the feather is lighter than the book",
    "the car is faster than the bike",
    "the snail is slower than the rabbit",
    "the cup is full", "the cup is empty",
    "the pillow is soft", "the rock is hard",
    "the drum is loud", "the whisper is quiet",
    "the sun is bright", "the closet is dark",
    "i have more blocks", "i have less blocks",
    "many birds fly", "few birds stay",

    # Weather variations
    "today is sunny", "today is cloudy",
    "today is rainy", "today is snowy",
    "today is windy", "today is stormy",
    "when it rains we use an umbrella",
    "when it snows we wear a coat",
    "when it is sunny we play outside",
    "when it is windy we fly a kite",

    # Season variations
    "spring is warm and flowers grow",
    "summer is hot and we swim",
    "fall is cool and leaves change",
    "winter is cold and we play in snow",

    # Plant variations
    "the seed needs water to grow",
    "the root holds the plant",
    "the stem carries water up",
    "the leaf makes food from sun",
    "the flower makes seeds",
    "the tree has roots stem leaves and branches",

    # Time variations
    "i eat breakfast in the morning",
    "i eat lunch in the afternoon",
    "i eat dinner at night",
    "i go to school today",
    "we will play tomorrow",
    "we played yesterday",
    "monday is the first day",
    "friday is the last school day",
    "saturday and sunday are the weekend",

    # Family variations
    "my mom loves me", "my dad loves me",
    "my sister plays with me", "my brother helps me",
    "my grandma tells stories", "my grandpa teaches me",
    "families are different", "all families love each other",

    # Community variations
    "the teacher reads to us",
    "the doctor checks my health",
    "the firefighter drives a big truck",
    "the police officer helps people",
    "the librarian finds books for me",
    "the mail carrier brings letters",
    "the farmer grows our food",
    "the baker makes bread",

    # Senses variations
    "i see colors", "i see shapes",
    "i hear music", "i hear birds",
    "i smell flowers", "i smell food",
    "i taste sweet", "i taste sour",
    "i feel soft", "i feel rough",

    # Health variations
    "i wash my hands before eating",
    "i brush my teeth morning and night",
    "i wear a helmet when i ride",
    "i buckle my seatbelt in the car",
    "i look both ways before crossing",
    "i stop at red and go at green",
    "i do not touch hot things",
    "i am careful with sharp things",

    # Movement variations
    "i run to the park", "i jump on the trampoline",
    "i hop like a bunny", "i skip down the street",
    "i climb the playground", "i throw the ball",
    "i catch the ball", "i kick the ball",
    "i balance on one foot", "i crawl like a baby",
    "i slide down the slide", "i swing on the swing",

    # Music variations
    "i sing a happy song", "i dance to the music",
    "i clap my hands", "i stomp my feet",
    "the music is fast", "the music is slow",
    "the music is loud", "the music is quiet",
    "i draw with crayons", "i paint with brushes",
    "i cut with scissors", "i paste on paper",
    "i color a picture", "i make something new",

    # Social variations
    "my name is important", "i am special",
    "i can do it by myself", "i try hard",
    "i wait for my turn", "i share my toys",
    "i help my friend", "i say i am sorry",
    "i use gentle hands", "i use walking feet",
    "i clean up my toys", "i line up quietly",
    "i raise my hand", "i listen with my ears",

    # Living vs non-living variations
    "a dog grows", "a cat needs food",
    "a tree breathes", "a flower moves toward sun",
    "a rock does not grow", "a toy does not eat",
    "a car does not breathe", "a book does not grow",
    "living things need water", "living things need air",

    # Money variations
    "a penny is copper", "a nickel is silver",
    "a dime is small", "a quarter is big",
    "i save my money", "i count my coins",
    "one penny two pennies three pennies",

    # Pretend play variations
    "i pretend to be a doctor", "i pretend to be a teacher",
    "i pretend to cook food", "i pretend to fly a plane",
    "we play house together", "we play store together",
    "we use our imagination", "pretend play is fun",
]


FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES
