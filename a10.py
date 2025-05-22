import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match

previous_name = ""


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)

    return match.group("radius")


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)

    return match.group("birth")


# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.


def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]

def incumbency_start(matches: List[str]) -> List[str]:
    """Returns incumbency start date of named person in matches

    Args:
        matches - match from pattern of person's name to find incumbency start date of

    Returns:
        incumbency start date of named person
    """
    return [get_incumbency_start(" ".join(matches))]

def incumbency_start_year(matches: List[str]) -> List[str]:
    """Returns incumbency start year of named person in matches

    Args:
        matches - match from pattern of person's name to find incumbency start year of

    Returns:
        incumbency start year of named person
    """
    date = get_incumbency_start(" ".join(matches))
    return [date.split(",")[1].strip() if date else "No answers"]

def incumbency_end(matches: List[str]) -> List[str]:
    """Returns incumbency end date of named person in matches

    Args:
        matches - match from pattern of person's name to find incumbency end date of

    Returns:
        incumbency end date of named person
    """
    return [get_incumbency_end(" ".join(matches))]

def incumbency_end_year(matches: List[str]) -> List[str]:
    """Returns incumbency end year of named person in matches

    Args:
        matches - match from pattern of person's name to find incumbency end year of

    Returns:
        incumbency end year of named person
    """
    date = get_incumbency_end(" ".join(matches))
    return [date.split(",")[1].strip() if date else "No answers"]

def number(matches: List[str]) -> List[str]:
    """Returns presidential number of named person in matches

    Args:
        matches - match from pattern of person's name to find presidential number of

    Returns:
        presidential number of named person
    """
    results = get_number(" ".join(matches))
    if len(results) > 1:
        return [results[0] + " & " + results[1]]
    return results


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]


def get_incumbency_start(name: str) -> str:
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"President of the United StatesIn office+(?P<officestart>[A-Za-z]+\s+\d{1,2},\s+\d{4})"
    error_text = "Page infobox has no incumbency start information"
    try:
        match = get_match(infobox_text, pattern, error_text)
        return match.group("officestart")
    except AttributeError:
        newpattern = r"IncumbentAssumed office\s+(?P<officestart>[A-Za-z]+\s+\d{1,2},\s+\d{4})"
        match = get_match(infobox_text, newpattern, error_text)
        return match.group("officestart")


def get_incumbency_end(name: str) -> str:
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"President of the United StatesIn office+([A-Za-z]+\s+\d{1,2},\s+\d{4})\s+(?P<officeend>[A-Za-z]+\s+\d{1,2},\s+\d{4})"

    error_text = "Page infobox has no incumbency end information"
    try:
        match = get_match(infobox_text, pattern, error_text)
        return match.group("officeend")
    except AttributeError:
        newpattern = r"IncumbentAssumed office\s+(?P<officestart>[A-Za-z]+\s+\d{1,2},\s+\d{4})"
        match = get_match(infobox_text, newpattern, error_text)
        match.group("officestart")
        return "Current incumbent"

def get_number(name: str) -> List[str]:
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(\d{2})(?=(?:st|nd|rd|th)\b(?: & \b\d+(?:st|nd|rd|th)\b)? President of the United States)"

    error_text = "Page infobox has no presidential number information"
    matches = re.findall(pattern, infobox_text)
    if not matches:
        raise AttributeError("Page infobox has no presidential number information")
    return matches



# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[Pattern, Action]] = [
    ("when did % take office".split(), incumbency_start),
    ("when did % become president".split(), incumbency_start),
    ("what year did % take office".split(), incumbency_start_year),
    ("what year did % become president".split(), incumbency_start_year),
    ("when did % begin his presidency".split(), incumbency_start),
    ("what year did % begin his presidency".split(), incumbency_start_year),
    ("when did % leave office".split(), incumbency_end),
    ("when did % end his presidency".split(), incumbency_end),
    ("what year did % leave office".split(), incumbency_end_year),
    ("when year did % end his presidency".split(), incumbency_end_year),
    ("what number president is %".split(), number),
    ("what number president was %".split(), number),
    ("which number president is %".split(), number),
    ("which number president was %".split(), number),

    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    global previous_name
    for pat, act in pa_list:
        mat:List[str] = match(pat, src)
        if mat is not None and "he" in mat or "they" in mat and previous_name is not None:
            answer = act([previous_name])
            return answer if answer else ["No answers"]
        if mat is not None:
            answer = act(mat)
            previous_name = mat[0]
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Presidental Information System")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
# query_loop()


# I included a feature to hold context between queries. It simply stores the last entered name in a global variable and calls upon it when the prononu 'he' is used
#demo:

answers = search_pa_list("when did abraham lincoln take office".replace("?", "").lower().split())
for ans in answers:
    print(ans)

answers = search_pa_list("when did he leave office".replace("?", "").lower().split())
for ans in answers:
    print(ans)