"""An object's name, in words that cannot carry a person's or a place's name, for files a company reads.

The site's own words never go into the company's files: only the WebApp can remove the site's name, address and
contact details, and no one can remove the name of a person the site mentions ("move Dr. Ana Patel's sample tray to
the fridge"). The build plan's object names and varieties (``website_build_plan``) are a model's short phrases from
those same words, so they are not shipped as written either. A phrase ships only as plain object words:

- a possessor is dropped with its possessive: the word that ends in ``'s`` (or ``'``), and the capitalised words and
  titles just before it ("Dr. Ana Patel's sample tray" is "sample tray");
- every word left must then be a word of ``WORDS`` (a closed list of plain object nouns, materials, colours, sizes,
  counts and small joining words, with their plurals), a count of up to four digits, or a unit of measure; and no word
  after the first may be capitalised, since a capital inside a phrase is a name's;
- otherwise nothing of the phrase ships, and the caller says what the object is in its own words: its parametric
  shape, or its role in the job.

What ships is rebuilt from the list's own words, in lower case, so no spelling of the site's reaches the file. Plain
code, no model: a phrase is shipped or not by these rules alone. A list word that is also someone's name ("Rose")
ships only alone at the start of a phrase, where it reads as the plain word and names no one.
"""
from __future__ import annotations

import re

__all__ = ["UNITS", "WORDS", "plain_phrase"]

#: The closed list. No given names, surnames, titles, street words or place names; colours that are also common given
#: names (rose, violet, ruby, amber, olive, hazel, ...) are left out on purpose.
WORDS = frozenset("""
a an the of with and or for on in into onto inside under below over above from to at by its each per one two three
four five six seven eight nine ten eleven twelve dozen several many few some all any both other another more up about
around roughly approximately single double pair set stack pile row group batch handful load lot time times once twice
whole part parts piece pieces kind kinds type types variety varieties size sizes shape shapes colour colours color
colors style styles model models mixed assorted various different same similar matching identical
small large big medium tiny huge tall short long wide narrow thin thick flat round square rectangular cylindrical
heavy light empty full half open closed opened sealed upper lower top bottom middle centre center front back rear
left right inner outer near far nearest first second third last next main spare extra new used clean dirty wet dry
hot cold frozen fresh loose stacked folded rolled wrapped boxed bagged packed labelled labeled numbered sorted
lidded handled wheeled portable mobile fixed standing hanging mounted rolling sliding hinged folding stacking
plastic metal metallic steel stainless aluminium aluminum iron wooden wood glass paper cardboard corrugated fabric
cloth rubber foam ceramic porcelain silicone wire mesh clear transparent white black grey gray red blue green
yellow orange purple pink brown silver gold beige tan dark pale coloured colored
box bin tote tray crate carton case container package parcel bag sack pouch envelope basket bucket pail barrel drum
keg can tin jar bottle flask jug pitcher canister tube vial ampoule cartridge capsule pot pan lid cap cover bowl
plate dish cup mug tumbler beaker cylinder pallet skid tub trough hamper caddy holder organiser organizer rack shelf
shelving drawer cabinet cupboard locker cubby wrapper wrap film roll spool reel coil cassette cooler packaging
table desk bench workbench counter countertop worktop surface stool chair seat cart trolley dolly stand pedestal
platform frame door gate hatch window panel fridge refrigerator freezer oven microwave dishwasher washer dryer sink
faucet tap basin toilet machine printer scanner conveyor belt station dock bay floor ground wall corner side edge
spot area zone region place position location slot opening hole hook peg rail rod bar pole post hanger handle knob
lever switch button dial valve pedal latch lock hinge kitchen lab laboratory office warehouse storeroom pantry
garage workshop storage shipping receiving loading delivery
fork knife spoon spatula ladle whisk tongs utensil cutlery napkin towel sponge brush mop broom dustpan trash garbage
waste recycling compost food fruit vegetable apple banana bread bun loaf egg milk water juice soda drink coffee tea
sugar flour rice pasta cereal snack sandwich meat fish cheese butter sauce soup produce ingredient spice bean potato
tomato onion lettuce carrot
sample specimen slide pipette tip well petri reagent rack centrifuge incubator microscope syringe swab glove mask
gown kit test cuvette block label barcode sharps biohazard medication medicine pill blister pack dressing bandage
gauze instrument tool device monitor
document file folder binder book notebook magazine mail letter card ticket receipt pen pencil marker stapler tape
scissors toner cable cord charger laptop computer keyboard mouse phone tablet screen display remote battery item
product goods merchandise stock inventory order shipment garment clothing clothes shirt t-shirt pants trousers
jacket shoe sock linen sheet blanket pillow laundry
wrench hammer screwdriver drill saw pliers screw bolt nut nail fastener fitting pipe hose bracket board plank beam
brick tile component forklift
toy ball game cube doll plant flower seed soil
""".split())
#: Units of measure: only right after a count ("500 ml"), so a lone letter is never shipped.
UNITS = frozenset("""
ml l litre liter g kg gram kilogram cm mm m metre meter inch inches oz lb lbs gallon
""".split())

_POSSESSIVE = re.compile(r".+(?:['’]s|s['’])$")
_COUNT = re.compile(r"\d{1,4}(?:\.\d{1,3})?")
_EDGE = "()[]{}\"“”‘’.;:!?"


def _listed(word: str, listed: frozenset[str] = WORDS) -> bool:
    """A word of ``listed``, as written or as the plural of one."""
    candidates = {word}
    if word.endswith("s"):
        candidates.add(word[:-1])
    if word.endswith("es"):
        candidates.add(word[:-2])
    if word.endswith("ies"):
        candidates.add(word[:-3] + "y")
    if word.endswith("ves"):
        candidates.update({word[:-3] + "f", word[:-3] + "fe"})
    return bool(candidates & listed)


def _part(word: str) -> bool:
    return bool(_COUNT.fullmatch(word)) or _listed(word)


def plain_phrase(text: object) -> str | None:
    """``text`` in plain object words, lower case, or None when any word of it is not one (see the module note).

    Besides ``WORDS``: a count of up to four digits, never right after another (no phone or street number), a unit
    of ``UNITS`` only right after a count, and a hyphenated word whose every part is one ("96-well"). A one-letter
    word written with a full stop is an initial, never a word."""
    if not isinstance(text, str):
        return None
    kept: list[tuple[str, bool]] = []
    for token in text.split():
        if _POSSESSIVE.fullmatch(token.rstrip(",;:.")):
            # The possessor goes with its possessive: the capitalised words and titles ("Dr.") just before it.
            while kept and (kept[-1][0][:1].isupper() or kept[-1][0].endswith(".")):
                kept.pop()
            continue
        kept.append((token, token.endswith(",")))
    words: list[str] = []
    previous_count = False
    for index, (token, comma) in enumerate(kept):
        core = token.strip(_EDGE + ",")
        if (not core or (index and any(character.isupper() for character in core))
                or (len(core) == 1 and token.rstrip(",").endswith("."))):
            return None
        word = core.lower()
        count = bool(_COUNT.fullmatch(word))
        if count and previous_count:
            return None
        if not (count or _listed(word) or (previous_count and _listed(word, UNITS))
                or ("-" in word and all(part and _part(part) for part in word.split("-")))):
            return None
        previous_count = count
        words.append(word + ("," if comma and index < len(kept) - 1 else ""))
    return " ".join(words) or None
