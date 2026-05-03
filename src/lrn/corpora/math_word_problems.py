"""
Math Word Problem Corpus - Addition, subtraction, multiplication, division,
fractions, ratios, percentages, multi-step problems with solution patterns
"""

ORIGINAL_CORPUS = [
    # Addition word problems
    "john has three apples and mary gives him five more how many apples does john have now",
    "there are twelve students in the classroom and eight more arrive how many students are there now",
    "a farmer has twenty four cows and buys fifteen more how many cows does the farmer have",
    "sarah saved twenty three dollars and her grandmother gave her seventeen dollars how much money does sarah have now",
    "a book has one hundred fifty six pages and you read eighty nine pages how many pages are left to read",

    # Subtraction word problems
    "there were thirty five birds on the tree and twelve flew away how many birds are left",
    "tom had fifty dollars and spent twenty eight dollars how much money does tom have left",
    "a tank holds two hundred liters of water and one hundred thirty five liters have been used how much water is left",
    "a pizza has eight slices and three slices were eaten how many slices are left",
    "the temperature was twenty five degrees and dropped by twelve degrees what is the temperature now",

    # Multiplication word problems
    "each box contains six pencils and there are eight boxes how many pencils are there in total",
    "a car travels at sixty miles per hour how far does it travel in four hours",
    "a classroom has five rows of desks with seven desks in each row how many desks are there",
    "a baker makes twelve dozen cookies how many cookies does the baker make",
    "each student needs three notebooks and there are twenty four students how many notebooks are needed",

    # Division word problems
    "forty eight candies are shared equally among six children how many candies does each child get",
    "a teacher has seventy two books and wants to put them on shelves with nine books each how many shelves are needed",
    "one hundred twenty students are divided into eight equal groups how many students are in each group",
    "a rope is fifty six meters long and is cut into seven equal pieces how long is each piece",
    "ninety six apples are packed into boxes of twelve how many boxes are needed",

    # Fraction word problems
    "a pizza is cut into eight equal slices and you eat three slices what fraction of the pizza did you eat",
    "a recipe calls for three quarters of a cup of sugar if you want to make half the recipe how much sugar do you need",
    "two thirds of the class are boys and there are thirty students how many boys are there",
    "a tank is five sixths full and one quarter of the water is used how much water is left",
    "she read two fifths of a book on monday and one fifth on tuesday what fraction of the book has she read",

    # Percentage word problems
    "a shirt costs forty dollars and is on sale for twenty five percent off what is the sale price",
    "there are two hundred students and sixty percent are girls how many girls are there",
    "a test has fifty questions and you got forty two correct what percentage did you score",
    "a store increased the price of an item by ten percent if the original price was thirty dollars what is the new price",
    "a population of one thousand grew by fifteen percent what is the new population",

    # Ratio word problems
    "the ratio of boys to girls in a class is three to two if there are thirty students how many boys are there",
    "a recipe uses flour and sugar in a ratio of four to one if you use two cups of sugar how much flour do you need",
    "the ratio of red marbles to blue marbles is five to three if there are forty marbles total how many are red",
    "a map has a scale of one to fifty thousand if two cities are three centimeters apart on the map how far apart are they in real life",
    "the ratio of cats to dogs in a shelter is two to five if there are thirty five dogs how many cats are there",

    # Multi-step word problems
    "john bought three shirts at twelve dollars each and two pairs of pants at twenty dollars each how much did he spend in total",
    "a rectangular garden is twelve meters long and eight meters wide what is the area and what is the perimeter",
    "a train travels at sixty kilometers per hour for two hours and then at eighty kilometers per hour for three hours what is the total distance",
    "a store sells apples for two dollars per pound and oranges for three dollars per pound if you buy four pounds of apples and three pounds of oranges how much do you pay",
    "a water tank is filled at a rate of five liters per minute and drained at a rate of two liters per minute how much water is added in one hour",

    # Algebra word problems
    "if x plus five equals twelve what is the value of x",
    "if three times a number equals twenty one what is the number",
    "if twice a number plus four equals fourteen what is the number",
    "the sum of two numbers is twenty five and their difference is seven what are the two numbers",
    "a number multiplied by itself equals one hundred forty four what is the number",
]

VARIED_EXAMPLES = [
    "a class has twenty eight students and fourteen are absent how many students are present",
    "a bus can carry forty five passengers and there are thirty two on board how many more can it carry",
    "each pack contains ten markers and there are fifteen packs how many markers are there",
    "one hundred twenty five cookies are shared among five friends how many does each get",
    "a cake is cut into twelve pieces and you eat four pieces what fraction remains",
    "a jacket costs eighty dollars and is discounted by thirty percent what is the final price",
    "the ratio of teachers to students is one to twenty five if there are five hundred students how many teachers are there",
    "mary saved fifteen dollars per week for eight weeks how much did she save",
    "a rectangle has length fifteen and width eight what is the area",
    "if two times a number minus three equals eleven what is the number",
]

FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES
