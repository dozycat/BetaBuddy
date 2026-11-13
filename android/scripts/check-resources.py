#!/usr/bin/env python3
"""Checks the Android resources and manifest without an Android SDK.

    python3 android/scripts/check-resources.py

- every XML file under res/ and the manifest is well formed
- the five languages have the same string names, placeholders and line breaks, no blank
  text, no unescaped apostrophe, no stray percent sign
- no string name is defined in two files of the same language
- every R.string / R.plurals / R.raw / R.drawable / R.mipmap / R.xml / R.color the code or
  the XML refers to exists, and no string is left unused
- locales_config.xml lists exactly the languages that have resources
- the manifest declares no permission, removes the ones dependencies merge in, and both
  backup sections of data_extraction_rules.xml exclude every domain
- the third-party license texts are in assets/licenses

Exits non-zero and prints each problem when something is wrong.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
RES = APP / "src/main/res"
ANDROID = "{http://schemas.android.com/apk/res/android}"
TOOLS = "{http://schemas.android.com/tools}"
LANGUAGES = {"en": "values", "zh-Hans": "values-b+zh+Hans", "zh-Hant": "values-b+zh+Hant", "ja": "values-ja", "ko": "values-ko"}
PLACEHOLDER = re.compile(r"%\d\$[sd]")
problems = []


def problem(text):
    problems.append(text)


def parse(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as error:
        problem(f"{path.relative_to(APP)}: not well-formed XML ({error})")
        return None


def strings(folder):
    """name (or name#quantity) -> (text, file name)"""
    found = {}
    for path in sorted((RES / folder).glob("strings*.xml")):
        root = parse(path)
        if root is None:
            continue
        for element in root:
            name = element.get("name")
            if element.tag == "string":
                items = [(name, element)]
            elif element.tag == "plurals":
                items = [(f"{name}#{item.get('quantity')}", item) for item in element.findall("item")]
            else:
                continue
            for key, node in items:
                if key in found:
                    problem(f"{folder}: {key} is defined in both {found[key][1]} and {path.name}")
                found[key] = ("".join(node.itertext()), path.name)
    return found


# 1. Well-formed XML everywhere.
for path in sorted(RES.rglob("*.xml")):
    parse(path)
manifest = parse(APP / "src/main/AndroidManifest.xml")

# 2. Five languages in step.
tables = {language: strings(folder) for language, folder in LANGUAGES.items()}
english = tables["en"]
untranslated = {"app_name"}   # translatable="false", default only
for language, table in tables.items():
    expected = {key for key in english if not key.endswith("#one")} - (untranslated if language != "en" else set())
    actual = {key for key in table if not key.endswith("#one")}
    for key in sorted(expected - actual):
        problem(f"{language}: missing {key}")
    for key in sorted(actual - expected):
        problem(f"{language}: {key} is not in the English file")
    for key, (text, _) in table.items():
        reference = english.get(key) or english.get(key.split("#")[0] + "#other")
        if reference is None:
            continue
        if not text.strip():
            problem(f"{language}: {key} is blank")
        if sorted(PLACEHOLDER.findall(text)) != sorted(PLACEHOLDER.findall(reference[0])):
            problem(f"{language}: {key} placeholders differ from English")
        if text.count("\\n") != reference[0].count("\\n"):
            problem(f"{language}: {key} line breaks differ from English")
        if re.search(r"(?<!\\)'", text):
            problem(f"{language}: {key} has an unescaped apostrophe")
        if "%" in PLACEHOLDER.sub("", text):
            problem(f"{language}: {key} has a stray percent sign")
for language in ("en", "ko"):
    for key, (text, _) in tables[language].items():
        if any("一" <= character <= "鿿" for character in text):
            problem(f"{language}: {key} still contains Chinese characters")
for language in LANGUAGES:
    if tables[language].get("app_language_code", ("",))[0] != language:
        problem(f"{language}: app_language_code must be {language}")

# 3. References from code and XML resolve; no string is unused.
defined = {"string": {key.split("#")[0] for key in english if "#" not in key},
           "plurals": {key.split("#")[0] for key in english if "#" in key},
           "raw": set(), "drawable": set(), "mipmap": set(), "xml": set(), "color": set(), "style": set()}
for folder in RES.iterdir():
    kind = folder.name.split("-")[0]
    if kind in ("raw", "drawable", "mipmap", "xml"):
        defined[kind].update(path.stem for path in folder.iterdir())
    elif kind == "values":
        for path in folder.glob("*.xml"):
            root = parse(path)
            for element in root if root is not None else []:
                if element.tag in ("color", "style"):
                    defined[element.tag].add(element.get("name").replace(".", "_"))
used = {kind: set() for kind in defined}
sources = list((APP / "src").rglob("*.kt"))
for path in sources:
    for kind, name in re.findall(r"\bR\.(\w+)\.(\w+)", path.read_text(encoding="utf-8")):
        if kind in used:
            used[kind].add(name)
            if name not in defined[kind]:
                problem(f"{path.relative_to(APP)}: R.{kind}.{name} does not exist")
for path in list(RES.rglob("*.xml")) + [APP / "src/main/AndroidManifest.xml"]:
    for kind, name in re.findall(r"@(\w+)/([\w.]+)", path.read_text(encoding="utf-8")):
        name = name.replace(".", "_")
        if kind in used:
            used[kind].add(name)
            if name not in defined[kind]:
                problem(f"{path.relative_to(APP)}: @{kind}/{name} does not exist")
for kind in ("string", "plurals"):
    for name in sorted(defined[kind] - used[kind]):
        problem(f"{kind} {name} is not used anywhere")

# 4. locales_config lists exactly the languages that have resources.
config = parse(RES / "xml/locales_config.xml")
if config is not None:
    listed = {element.get(ANDROID + "name") for element in config}
    if listed != set(LANGUAGES):
        problem(f"locales_config.xml lists {sorted(listed)}, resources exist for {sorted(LANGUAGES)}")
for folder in (RES / "raw").iterdir():
    for language_folder in ("raw-b+zh+Hans", "raw-b+zh+Hant"):
        if not (RES / language_folder / folder.name).is_file():
            problem(f"{language_folder}/{folder.name} is missing")

# 5. Manifest: no permission, nothing online, nothing backed up.
if manifest is not None:
    removed = set()
    for element in manifest.findall("uses-permission"):
        name = element.get(ANDROID + "name")
        if element.get(TOOLS + "node") == "remove":
            removed.add(name)
        else:
            problem(f"manifest declares permission {name}")
    for name in ("android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE", "android.permission.WAKE_LOCK"):
        if name not in removed:
            problem(f"manifest does not remove {name} merged in from dependencies")
    application = manifest.find("application")
    if application.get(ANDROID + "allowBackup") != "false" or application.get(ANDROID + "fullBackupContent") != "false":
        problem("manifest must set allowBackup and fullBackupContent to false")
    if application.get(ANDROID + "usesCleartextTraffic") != "false":
        problem("manifest must set usesCleartextTraffic to false")
rules = parse(RES / "xml/data_extraction_rules.xml")
if rules is not None:
    domains = {"root", "file", "database", "sharedpref", "external"}
    for section in ("cloud-backup", "device-transfer"):
        node = rules.find(section)
        excluded = {element.get("domain") for element in node.findall("exclude")} if node is not None else set()
        if excluded != domains or (node is not None and node.findall("include")):
            problem(f"data_extraction_rules.xml: <{section}> must exclude {sorted(domains)} and include nothing")

# 6. The third-party license texts are bundled.
for name in ("components.txt", "Apache-2.0.txt", "MediaPipe-NOTICE.txt"):
    path = APP / "src/main/assets/licenses" / name
    if not path.is_file() or path.stat().st_size == 0:
        problem(f"assets/licenses/{name} is missing or empty")

if problems:
    print("\n".join(problems))
    sys.exit(1)
total = len({key for key in english})
print(f"Resources OK: {total} strings in {len(LANGUAGES)} languages, {len(sources)} Kotlin files checked for references.")
