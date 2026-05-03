"""
Computer Logic and Programming Foundations - Grades 4 through 8
Introduces algorithms, data structures, and programming concepts.

Grade 4-5: Algorithms, debugging, functions, arrays/lists
Grade 6-8: Objects, classes, Boolean logic, complexity
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


LESSONS_4TH = [
    {
        "title": "Algorithms: Step-by-Step Solutions",
        "concept": "An algorithm is a clear set of steps to solve a problem.",
        "sentences": [
            "to find the largest number compare each number to the current largest",
            "to sort items arrange them from smallest to biggest step by step",
            "to search a list check each item until you find what you want",
            "to add numbers align them by place value then add each column",
            "to multiply break the problem into easier parts",
            "to find average add all numbers then divide by how many there are",
            "to find mode count how many times each value appears",
            "to find median arrange values in order then pick the middle one",
            "to calculate range subtract the smallest from the largest",
            "to solve problems break them into smaller manageable parts",
        ],
    },
    {
        "title": "Debugging: Finding and Fixing Errors",
        "concept": "Debugging is the process of finding and fixing mistakes in code.",
        "sentences": [
            "the program crashed because of infinite loop",
            "the answer is wrong because of off by one error",
            "the variable has wrong value because assignment was reversed",
            "the function returns none because return statement is missing",
            "the loop never ends because the increment is missing",
            "the array index is out of bounds because size calculation was wrong",
            "the string compare failed because case was different",
            "the file did not open because the path was wrong",
            "the output is garbled because encoding was wrong",
            "the crash happened at line forty seven where variable was used",
        ],
    },
    {
        "title": "Functions: Reusable Blocks",
        "concept": "Functions are named blocks of code that perform specific tasks.",
        "sentences": [
            "function calculate area takes length and width and returns product",
            "function find maximum takes three numbers and returns the biggest",
            "function check parity takes number and returns true if even",
            "function convert celsius takes temperature and returns fahrenheit",
            "function validate input takes string and returns true if valid",
            "function generate random chooses number between min and max",
            "function format date takes day month year and returns formatted string",
            "function calculate tax takes amount and rate and returns total",
            "function reverse string takes text and returns backwards version",
            "function count vowels takes word and returns how many vowels it has",
        ],
    },
    {
        "title": "Arrays and Lists",
        "concept": "Arrays and lists store multiple values in order.",
        "sentences": [
            "the list holds five student names in order",
            "access element at index three to get the fourth item",
            "add new item to end of list using append",
            "remove item from list using delete at index",
            "loop through list to process each element",
            "the array stores ten numbers in consecutive memory locations",
            "sort the list from lowest to highest value",
            "search the list to find if a value exists",
            "the two dimensional array has rows and columns like a grid",
            "filter the list to keep only items that match condition",
        ],
    },
]

LESSONS_5TH = [
    {
        "title": "Boolean Logic: AND OR NOT",
        "concept": "Boolean logic combines conditions using AND, OR, and NOT.",
        "sentences": [
            "if age is greater than twelve and height is above four feet then can ride",
            "if user is admin or user is owner then grant access",
            "if not logged in then redirect to login page",
            "if temperature is above one hundred and pressure is high then warning sounds",
            "if user entered correct password and correct username then login succeeds",
            "if item is in stock and price is affordable then can purchase",
            "if response is yes or response is maybe then proceed",
            "if not error and not timeout then continue processing",
            "if first condition is true and second condition is true then both must be met",
            "if either option is selected or the other option is selected then one is chosen",
        ],
    },
    {
        "title": "Variables and Data Types",
        "concept": "Different types of data need different kinds of storage.",
        "sentences": [
            "integer stores whole numbers like twelve or negative five",
            "float stores decimal numbers like three point one four one five",
            "string stores text like hello world",
            "boolean stores true or false only",
            "character stores single letter like a or b",
            "array stores multiple items of the same type",
            "variable name should describe what it holds",
            "change the variable value using assignment operator",
            "convert string to number before adding",
            "check the data type before performing operation",
        ],
    },
    {
        "title": "Loops: For and While",
        "concept": "Loops repeat code until a condition is met.",
        "sentences": [
            "for each item in the list process it",
            "while the counter is less than ten increment and continue",
            "for i from zero to nine iterate ten times",
            "while not at end of file read next line",
            "for each student in the class calculate their average",
            "while the puzzle is not solved try another approach",
            "repeat until the answer is correct",
            "loop through the array from start to finish",
            "while the game is running check for user input",
            "for count in range five repeat the action five times",
        ],
    },
    {
        "title": "Pseudo-code: Planning Before Coding",
        "concept": "Pseudo-code is plain language description of code logic.",
        "sentences": [
            "start with input get the number from user",
            "next process convert fahrenheit to celsius using formula",
            "then output display the result on screen",
            "if condition is met then do this otherwise do that",
            "loop until reaching the end of data",
            "initialize counter to zero before starting",
            "update counter by adding one each time through",
            "check if counter exceeds limit then stop",
            "divide the problem into three main parts",
            "combine results from each part at the end",
        ],
    },
]

LESSONS_6TH = [
    {
        "title": "Objects and Properties",
        "concept": "Objects have properties (attributes) and behaviors (methods).",
        "sentences": [
            "the car object has color property and speed property",
            "the student object has name property and grade property",
            "call the move method on the player object to change position",
            "set the x coordinate property to move the sprite horizontally",
            "the book object has title property and author property and page count property",
            "access the temperature property to get current reading",
            "update the health property when player takes damage",
            "the rectangle object has width property and height property",
            "call the draw method to render the shape on screen",
            "the game character has strength property and agility property",
        ],
    },
    {
        "title": "Classes and Instances",
        "concept": "A class is a blueprint; an instance is a specific object created from it.",
        "sentences": [
            "define a dog class with name and breed properties",
            "create instance named buddy from the dog class",
            "each dog instance has its own name and breed values",
            "the rectangle class defines width height color properties",
            "instantiate four rectangle objects for the game board",
            "the player class defines position velocity health properties",
            "spawn multiple enemy instances from the enemy class",
            "modify the instance properties without changing the class",
            "the button class defines click behavior and label property",
            "create submit button instance and cancel button instance",
        ],
    },
    {
        "title": "Events and Event Handlers",
        "concept": "Programs respond to events like clicks and keypresses.",
        "sentences": [
            "when user clicks the button event fires",
            "handle the key down event to move the player",
            "the mouse move event tracks cursor position",
            "trigger collision event when two objects touch",
            "handle the timer event to update the display",
            "respond to touch event on mobile devices",
            "the event handler processes the event and updates state",
            "queue events when they arrive faster than processing speed",
            "remove event listener when no longer needed",
            "custom events allow communication between components",
        ],
    },
    {
        "title": "Data Structures: Stacks and Queues",
        "concept": "Stacks and queues organize data in specific ways.",
        "sentences": [
            "push item onto top of stack",
            "pop item from top of stack last in first out",
            "the undo function uses a stack to remember actions",
            "enqueue item at the back of queue",
            "dequeue item from front of queue first in first out",
            "the print queue processes jobs in order received",
            "stack overflow happens when too many items are pushed",
            "check if queue is empty before dequeueing",
            "peek at top item without removing it from stack",
            "priority queue serves important items before others",
        ],
    },
]

LESSONS_7TH = [
    {
        "title": "Algorithm Complexity: Big O",
        "concept": "Big O notation describes how algorithm performance scales.",
        "sentences": [
            "constant time means one operation regardless of input size",
            "linear time means time grows directly with input size",
            "binary search is logarithmic time halving data each step",
            "merge sort is n log n time divide and conquer sorting",
            "nested loops create quadratic time performance",
            "hash table lookup is constant time on average",
            "tree traversal is logarithmic time for balanced trees",
            "bubble sort is quadratic time simple but slow",
            "the algorithm runs in constant space no matter the input",
            "optimize the inner loop to improve overall performance",
        ],
    },
    {
        "title": "Recursion: Functions Calling Themselves",
        "concept": "Recursion solves problems by breaking them into smaller versions of themselves.",
        "sentences": [
            "factorial function calls itself with smaller number until base case",
            "base case stops the recursion from continuing forever",
            "recursive function for Fibonacci adds previous two numbers",
            "tree traversal uses recursion to visit child nodes",
            "recursive search explores branches until finding target",
            "each recursive call adds a layer to the call stack",
            "tail recursion can be optimized to avoid stack growth",
            "recursive solution for tower of hanoi moves disks one by one",
            "memoization caches results to avoid repeated computation",
            "recursive backtracking explores possibilities and undoes choices",
        ],
    },
    {
        "title": "Searching and Sorting Algorithms",
        "concept": "Common algorithms for organizing and finding data.",
        "sentences": [
            "linear search checks each element until finding match",
            "binary search requires sorted data and halves range each step",
            "bubble sort swaps adjacent elements that are out of order",
            "insertion sort builds sorted list one element at a time",
            "merge sort divides array in half sorts each half merges results",
            "quick sort picks pivot partitions data around it",
            "selection sort finds minimum places it at start repeats",
            "hash search uses hash function to compute index directly",
            "stable sorting preserves order of equal elements",
            "in place sorting uses only constant extra memory",
        ],
    },
    {
        "title": "Debugging Advanced Problems",
        "concept": "Systematic approaches to finding and fixing complex bugs.",
        "sentences": [
            "reproduce the bug consistently before trying to fix it",
            "narrow down the problem area by adding print statements",
            "check the input values at each step of the algorithm",
            "log the state before and after each iteration",
            "compare working version with broken version to spot difference",
            "use binary search on the code to find where behavior changes",
            "test edge cases like empty input or very large values",
            "verify assumptions about what the code actually does versus what it should do",
            "clean up temporary debug code before finishing",
            "add regression test to prevent bug from returning",
        ],
    },
]

LESSONS_8TH = [
    {
        "title": "Object-Oriented Design",
        "concept": "OOD organizes code around objects and their interactions.",
        "sentences": [
            "encapsulation hides internal details from outside code",
            "inheritance allows new class to derive properties from parent",
            "polymorphism lets different objects respond to same message differently",
            "composition builds complex objects from simpler ones",
            "the dog class inherits from the animal class",
            "interface defines method signatures without implementation",
            "abstract class provides base methods for subclasses to override",
            "favor composition over inheritance for flexibility",
            "single responsibility means each class does one thing well",
            "design pattern provides reusable solution to common problem",
        ],
    },
    {
        "title": "APIs and Modular Design",
        "concept": "APIs define how components communicate without exposing internals.",
        "sentences": [
            "the api endpoint receives request and returns json response",
            "authentication token must be included in request header",
            "rate limiting prevents too many requests in short time",
            "error response includes status code and message",
            "version the api to maintain backwards compatibility",
            "document the input format and output format clearly",
            "use descriptive endpoint names following rest conventions",
            "cache responses to improve performance for repeated requests",
            "validate input before processing to prevent attacks",
            "log all api calls for debugging and analytics",
        ],
    },
    {
        "title": "Boolean Algebra and Logic Gates",
        "concept": "Logic gates implement Boolean operations in hardware.",
        "sentences": [
            "and gate outputs one only when all inputs are one",
            "or gate outputs one when any input is one",
            "not gate inverts the input signal",
            "nand gate outputs zero only when all inputs are one",
            "nor gate outputs one only when all inputs are zero",
            "xor gate outputs one when inputs differ",
            "combine gates to build half adder for binary addition",
            "truth table shows all possible input output combinations",
            "demorgans law transforms and to or with negation",
            "boolean expression can be simplified using algebraic rules",
        ],
    },
    {
        "title": "Problem Decomposition and Abstraction",
        "concept": "Breaking complex problems into manageable pieces with clear interfaces.",
        "sentences": [
            "abstract the problem to hide unnecessary details",
            "the interface specifies what operations are available not how they work",
            "decompose the system into modules with single responsibility",
            "information hiding protects internal state from outside access",
            "refine the abstraction until the interface is clean and complete",
            "each layer of abstraction builds on the layer below",
            "top down design starts with high level requirements",
            "bottom up design starts with reusable components",
            "iterate between levels of detail to refine the solution",
            "validate that decomposition matches the real problem structure",
        ],
    },
]


def get_programming_lessons(grade, full=False):
    """Get computer logic/programming lessons for a grade."""
    if grade == 4:
        return LESSONS_4TH[:4]
    elif grade == 5:
        return LESSONS_5TH[:4]
    elif grade == 6:
        return LESSONS_6TH[:4]
    elif grade == 7:
        return LESSONS_7TH[:4]
    elif grade == 8:
        return LESSONS_8TH[:4]
    return []


def get_all_programming_sentences(grade):
    """Get all programming-related sentences for a grade."""
    lessons = get_programming_lessons(grade)
    sentences = []
    for lesson in lessons:
        sentences.extend(lesson.get("sentences", []))
    return sentences