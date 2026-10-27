"""Read-only journal views over the single locked manuscript and SI sources.

The label helper deliberately has no document or project-data dependencies.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


_JOURNALS = {"science", "natcatal"}
_PROTECTED = re.compile(
    r"`[^`]*`|(?<=\]\()[^\n)]+(?=\))|https?://[^\s<>]+|"
    r"(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|[A-Za-z0-9_.-]+[\\/])[^\s<>`]*"
)
_S_LABEL = re.compile(
    r"\b(?P<label>fig(?:ure)?s?\.?|tables?)\s+(?P<body>S\d+[A-Ha-h]?"
    r"(?:\s*(?:,\s*(?:and\s+)?|and\s+|to\s+|[\u2013\u2014-])\s*S?\d+[A-Ha-h]?)*)",
    re.I,
)
_MAIN_LABEL = re.compile(
    r"\b(?P<label>Figs?\.|Figures?)\s+(?P<number>[1-4])"
    r"(?P<panels>[A-H](?:\s*(?:,\s*(?:and\s+)?|and\s+|to\s+|[\u2013-])\s*[A-H])*)"
)
_PAREN_REF = re.compile(r"\((?:[1-9]|[12]\d|3[0-6])(?:\s*[,\u2013]\s*(?:[1-9]|[12]\d|3[0-6]))*\)")
_EXPLICIT_REF = re.compile(r"\brefs?\.?\s+([0-9]+(?:\s*,\s*[0-9]+)*)")
_MAIN_TEXT_REF = re.compile(r"(?<=main-text ref\. )\d+")
_FIG_CAPTION = re.compile(r"^\*\*Fig\. S(\d+) \|")
_MAIN_CAPTION = re.compile(r"^\*\*Fig\. ([1-4])\. ")
_SOURCE_MS = "01_MANUSCRIPT/Manuscript_Nature.md"
_SOURCE_SI = "03_SUPPLEMENTARY_INFORMATION/Supplementary_Information.md"
_INITIALS = r"(?:[A-Z]\.(?:-[A-Z]\.)?\s*)+"
_AUTHOR_FIRST = re.compile(r"(?P<initials>" + _INITIALS + r")\s+(?P<surname>[^,*]+?)(?P<et_al>\s+\*et al\.\*)?,\s+")
_SURNAME_FIRST = re.compile(r"(?P<surname>[^,.&*]+?),\s+(?P<initials>" + _INITIALS + r")(?:,\s+|\s+&\s+|\s+)")


def _journal(journal):
    if journal not in _JOURNALS:
        raise ValueError(f"Unknown journal: {journal!r}")


def _map_s_label(match):
    label, body = match.group("label", "body")
    numbers = [int(value) for value in re.findall(r"\d+", body)]
    plural = label.rstrip(".").lower().endswith("s") or len(numbers) > 1
    if label.lower().startswith("table"):
        noun = "Tables" if plural else "Table"
        return f"Supplementary {noun} " + re.sub(r"S(?=\d)", "", body)
    if min(numbers) <= 10 < max(numbers):
        interval = re.fullmatch(r"S(\d+)\s*([\u2013\u2014-]|to)\s*S?(\d+)", body)
        if interval:
            start, separator, end = interval.groups()
            separator = " to " if separator == "to" else separator
            return (f"Extended Data Figs. {start}{separator}10 and "
                    f"Supplementary Figs. 1{separator}{int(end) - 10}")
        raise ValueError(f"Mixed analysis/platform figure list needs explicit labels: {match.group(0)}")
    prefix = "Extended Data" if max(numbers) <= 10 else "Supplementary"
    offset = 0 if max(numbers) <= 10 else 10
    converted = re.sub(r"S?(\d+)", lambda m: str(int(m[1]) - offset), body)
    return f"{prefix} {'Figs.' if plural else 'Fig.'} {converted}"


def _format_unprotected(text):
    def main_label(match):
        preceding = text[:match.start()]
        if preceding.endswith(("Extended Data ", "Supplementary ")):
            return match.group(0)
        return f"{match['label']} {match['number']}{match['panels'].lower()}"

    text = _MAIN_LABEL.sub(main_label, text)
    text = re.sub(r"\b(Fig\. [1-4]), ([A-H]) (to|and) ([A-H])\b",
                  lambda m: f"{m[1]}, {m[2].lower()} {m[3]} {m[4].lower()}", text)
    return _S_LABEL.sub(_map_s_label, text).replace("Materials and Methods", "Methods")


def format_labels(text, journal="natcatal"):
    """Map prose labels without changing code spans, URLs or image/link paths."""
    _journal(journal)
    if journal == "science":
        return text
    out, position = [], 0
    for match in _PROTECTED.finditer(text):
        out.extend((_format_unprotected(text[position:match.start()]), match.group(0)))
        position = match.end()
    out.append(_format_unprotected(text[position:]))
    return "".join(out)


def _records(blocks, source):
    section, out = "", []
    for index, (kind, content) in enumerate(blocks):
        if kind == "h2":
            section = content
        out.append({"kind": kind, "content": deepcopy(content), "source": source,
                    "source_index": index, "id": f"{source}:block:{index}", "section": section})
    return out


def _generated(kind, content, identifier):
    return {"kind": kind, "content": content, "id": f"generated:{identifier}",
            "source": None, "source_index": None, "section": ""}


def _reference_ids(token):
    out = []
    for component in token.strip("()").split(","):
        if "\u2013" in component:
            start, end = map(int, component.split("\u2013"))
            out.extend(range(start, end + 1))
        else:
            out.append(int(component.strip()))
    return out


def _citation_spans(text, context):
    spans = []
    if context == "main":
        spans.extend((m.start(), m.end(), _reference_ids(m[0])) for m in _PAREN_REF.finditer(text))
    spans.extend((m.start(), m.end(), [int(m[0])]) for m in _MAIN_TEXT_REF.finditer(text))
    for match in _EXPLICIT_REF.finditer(text):
        if text[max(0, match.start() - 10):match.start()].endswith("main-text "):
            continue
        spans.append((match.start(), match.end(), _reference_ids(match[1])))
    spans.extend((m.start(), m.end(), [32]) for m in re.finditer("\u00b3\u00b2", text))
    return sorted(spans)


def _citations(text, context, main_map, local_map=None):
    out, position = [], 0
    for start, end, references in _citation_spans(text, context):
        if start < position:
            raise ValueError("Overlapping reference annotations")
        mapping = local_map if local_map is not None and all(number in local_map for number in references) else main_map
        if not all(number in mapping for number in references):
            raise ValueError(f"Unresolved reference in {text[start:end]!r}")
        out.extend((text[position:start].rstrip(), "<sup>" + ",".join(str(mapping[number]) for number in references) + "</sup>"))
        position = end
    out.append(text[position:])
    return "".join(out)


def _context(record):
    if record["source"] == _SOURCE_MS and record["section"] != "Figure legends":
        return "main"
    return "caption" if _FIG_CAPTION.match(str(record["content"])) else "supplementary"


def reference_facts(text):
    """Recover authors and the exact title/journal/volume/pages/year remainder."""
    body = re.sub(r"^\d+\.\s+", "", text)
    pattern = _AUTHOR_FIRST if _AUTHOR_FIRST.match(body) else _SURNAME_FIRST
    authors, position, et_al = [], 0, False
    while match := pattern.match(body, position):
        authors.append({"surname": match["surname"].strip(), "initials": " ".join(match["initials"].split())})
        et_al = et_al or bool(match.groupdict().get("et_al"))
        position = match.end()
    if not authors:
        raise ValueError(f"Reference author structure is not recognized: {text[:100]}")
    remainder = body[position:]
    et_al = et_al or bool(re.match(r"^\*et al\.\*", remainder))
    remainder = re.sub(r"^\*et al\.\*,?\s*", "", remainder)
    return {"authors": authors, "et_al": et_al, "remainder": remainder}


def nature_reference(text):
    facts = reference_facts(text)
    names = [f"{author['surname']}, {author['initials']}" for author in facts["authors"]]
    author_text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " & " + names[-1]
    if facts["et_al"]:
        author_text += " *et al.*"
    number = re.match(r"\d+\.", text)[0]
    result = number + " " + author_text + " " + facts["remainder"]
    if reference_facts(result) != facts:
        raise ValueError("Nature author reordering changes bibliographic facts")
    return result


def _render(record, main_map, local_map=None):
    out = deepcopy(record)
    content = out["content"]
    if out["kind"] == "img":
        out["content"] = (format_labels(content[0]), content[1])
    elif out["kind"] == "table":
        out["content"] = [format_labels(row) for row in content]
    else:
        content = format_labels(content)
        if out["section"] != "References":
            content = _citations(content, _context(record), main_map, local_map)
        if _MAIN_CAPTION.match(str(record["content"])):
            content = re.sub(r"\(([A-H])\)", lambda m: f"(**{m[1].lower()}**)", content)
        out["content"] = content
    return out


def _table_atoms(rows):
    for row_index, row in enumerate(rows):
        if row_index == 1:
            continue
        cells = [value.strip() for value in re.split(r"(?<!\\)\|", row.strip())[1:-1]]
        for column_index, cell in enumerate(cells):
            yield row_index, column_index, cell


def _key(destination, kind, content):
    return (destination, kind, json.dumps(content, ensure_ascii=True, sort_keys=True))


def _numbers(text, context):
    text = str(text)
    for start, end, _ in reversed(_citation_spans(text, context)):
        text = text[:start] + text[end:]
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"(?:Extended Data |Supplementary )?(?:Figs?\.|Figures?|Tables?)\s+S?\d+[A-Ha-h]?(?:\s*(?:[,\u2013-]|to|and)\s*S?\d+[A-Ha-h]?)*", "", text, flags=re.I)
    text = re.sub(r"Supplementary Note\s+\d+", "", text)
    return re.findall(r"[0-9]+(?:[.,][0-9]+)*", text)


def _scientific_text(text, context, main_caption=False, bibliography=False):
    text = str(text)
    if bibliography:
        return json.dumps(reference_facts(text), ensure_ascii=True, sort_keys=True)
    for start, end, _ in reversed(_citation_spans(text, context)):
        text = text[:start] + text[end:]
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"\b(Fig\. [1-4]), ([A-Ha-h]) (to|and) ([A-Ha-h])\b",
                  lambda m: f"{m[1]}, {m[2].lower()} {m[3]} {m[4].lower()}", text)
    text = re.sub(r"(?:Extended Data |Supplementary )?(?:Figs?\.|Figures?|Tables?)\s+S?\d+[A-Ha-h]?"
                  r"(?:\s*(?:[,\u2013-]|to|and)\s*S?\d+[A-Ha-h]?)*", "", text, flags=re.I)
    # Letter-only tails such as Fig. 1C,D are part of the main panel reference.
    text = re.sub(r"(?<=,)[A-H]\b", lambda m: m[0].lower(), text)
    if main_caption:
        text = re.sub(r"\(([A-Ha-h])\)", lambda m: f"({m[1].lower()})", text)
    text = text.replace("Materials and Methods", "Methods")
    text = text.replace("The data and code are available", "The data are available")
    text = re.sub(r"^Funding\.\s*", "", text)
    return re.sub(r"\s+", "", text)


def _atoms(destination, records):
    for record in records:
        if record["kind"] == "table":
            for row, column, content in _table_atoms(record["content"]):
                yield record, f":cell:{row}:{column}", "cell", content
        else:
            yield record, "", record["kind"], record["content"]


def _manifest(root, sources, outputs, required, original):
    items, assigned = [], []
    for destination, records in outputs.items():
        for record in records:
            if record["id"] in required:
                assigned.append(record["id"])
        for record, suffix, kind, expected in _atoms(destination, records):
            source = original.get(record["id"])
            original_content = source["content"] if source else None
            if kind == "cell" and source:
                originals = {(row, column): value for row, column, value in _table_atoms(original_content)}
                row, column = map(int, suffix.split(":")[-2:])
                original_content = originals[(row, column)]
            before = after = text_before = text_after = None
            if source and kind in {"p", "li", "cell"} and source["section"] != "References":
                before = _numbers(original_content, _context(source))
                after = _numbers(expected, _context(source))
            if source and kind in {"p", "li", "cell"}:
                main_caption = bool(_MAIN_CAPTION.match(str(source["content"])))
                bibliography = source["section"] == "References"
                text_before = _scientific_text(original_content, _context(source), main_caption, bibliography)
                text_after = _scientific_text(expected, _context(source), main_caption, bibliography)
            items.append({"id": record["id"] + suffix, "source_file": record["source"],
                          "source_block": record["source_index"], "destination": destination,
                          "kind": kind, "original_content": original_content, "expected_content": deepcopy(expected),
                          "scientific_numbers_before": before, "scientific_numbers_after": after,
                          "scientific_numbers_conserved": before == after,
                          "scientific_text_before": text_before, "scientific_text_after": text_after,
                          "scientific_text_conserved": text_before == text_after})
    return {"version": 1, "source_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
            "items": items, "required_source_ids": sorted(required), "assigned_source_ids": assigned}


def verify_documents(view):
    """Check the actual view against source-linked paragraphs, cells and images."""
    manifest = view["content_manifest"]
    expected = Counter(_key(item["destination"], item["kind"], item["expected_content"]) for item in manifest["items"])
    actual = Counter()
    for destination in ("manuscript", "supplementary"):
        records = [{"kind": kind, "content": content} for kind, content in view[destination]]
        for _, _, kind, content in _atoms(destination, records):
            actual[_key(destination, kind, content)] += 1
    for item in view["extended_data"]:
        actual[_key("extended_data", "img", (f"Extended Data Fig. {item['number']}", item["image_path"]))] += 1
        actual[_key("extended_data", "p", item["caption"])] += 1
    missing, extra = expected - actual, actual - expected
    assignments = Counter(manifest["assigned_source_ids"])
    unassigned = sorted(set(manifest["required_source_ids"]) - assignments.keys())
    duplicated = sorted(key for key, count in assignments.items() if count != 1)
    changed_numbers = [item["id"] for item in manifest["items"] if not item["scientific_numbers_conserved"]]
    changed_text = [item["id"] for item in manifest["items"] if not item["scientific_text_conserved"]]
    return {"status": "PASS" if not (missing or extra or unassigned or duplicated or changed_numbers or changed_text) else "FAIL",
            "checked_blocks_and_cells": len(manifest["items"]),
            "checked_table_cells": sum(item["kind"] == "cell" for item in manifest["items"]),
            "missing_atoms": sum(missing.values()), "unexpected_atoms": sum(extra.values()),
            "missing_source_ids": unassigned, "duplicate_source_ids": duplicated,
            "changed_scientific_numbers": changed_numbers, "changed_scientific_text": changed_text}


def documents(root, journal="natcatal"):
    """Return both document views without writing or relocking any source."""
    _journal(journal)
    import build_submission_package as builder

    root = Path(root).resolve()
    ms = _records(builder.blocks((root / _SOURCE_MS).read_text(encoding="utf-8")), _SOURCE_MS)
    si = _records(builder.blocks((root / _SOURCE_SI).read_text(encoding="utf-8")), _SOURCE_SI)
    original = {record["id"]: record for record in ms + si}
    reference_records = {int(re.match(r"(\d+)\. ", record["content"])[1]): record
                         for record in ms if record["section"] == "References" and record["kind"] == "p"}
    if journal == "science":
        outputs = {"manuscript": ms, "supplementary": si, "extended_data": []}
        refs = {"main": [{"number": number, "source_number": number, "text": record["content"]} for number, record in reference_records.items()],
                "supplementary": [], "mapping": {"main": {number: number for number in reference_records}, "supplementary": {}}}
        manifest = _manifest(root, (_SOURCE_MS, _SOURCE_SI), outputs, set(original), original)
        return _view(journal, outputs, [], refs, manifest)

    figure_records, removed = {}, set()
    for index, record in enumerate(si):
        if record["kind"] != "img":
            continue
        if index + 1 >= len(si):
            raise ValueError("A figure image has no following caption")
        caption = si[index + 1]
        match = _FIG_CAPTION.match(str(caption["content"]))
        if caption["kind"] != "p" or not match:
            raise ValueError("Every supplementary image must have its adjacent caption")
        number = int(match[1])
        if number in figure_records:
            raise ValueError(f"Duplicate figure S{number}")
        figure_records[number] = (record, caption)
        if number <= 10:
            removed.update((record["id"], caption["id"]))
    if set(figure_records) != set(range(1, 27)):
        raise ValueError("The current source must contain figures S1-S26 exactly once")
    methods = [record for record in si if record["section"] == "Materials and Methods"
               and record["kind"] != "h2" and record["id"] not in removed]
    note_start = next(index for index, record in enumerate(si) if record["kind"] == "h2" and record["content"].startswith("Supplementary Note 1 |"))
    notes = [record for record in si[note_start:] if record["id"] not in removed]
    main_body = [record for record in ms if record["section"] in {"", "Abstract", "Results", "Discussion"}]
    availability = [record for record in ms if record["section"] == "Data and materials availability" and record["kind"] == "p"]
    legends = [record for record in ms if record["section"] == "Figure legends" and record["kind"] == "p"]
    citation_order = main_body + methods + availability + legends + [figure_records[number][1] for number in range(1, 11)]
    ordered = []
    for record in citation_order:
        if record["kind"] not in {"p", "li"}:
            continue
        for _, _, numbers in _citation_spans(record["content"], _context(record)):
            for number in numbers:
                if number not in ordered:
                    ordered.append(number)
    main_map = {number: position + 1 for position, number in enumerate(ordered)}
    local_order = []
    for record in notes:
        if record["kind"] not in {"p", "li"}:
            continue
        for _, _, numbers in _citation_spans(record["content"], _context(record)):
            for number in numbers:
                if number not in main_map and number not in local_order:
                    local_order.append(number)
    local_map = {number: position + 1 for position, number in enumerate(local_order)}
    if set(main_map) | set(local_map) != set(reference_records):
        raise ValueError("The journal reference views must account for every source reference")

    render = lambda record: _render(record, main_map, local_map)
    main = [render(record) for record in main_body]
    main.append(_generated("h2", "Methods", "methods-heading"))
    main.extend(render(record) for record in methods)
    main.append(_generated("h2", "References", "main-references-heading"))
    refs = {"main": [], "supplementary": [], "mapping": {"main": main_map, "supplementary": local_map}}
    for scope, mapping in (("main", main_map), ("supplementary", local_map)):
        for source_number, number in mapping.items():
            record = deepcopy(reference_records[source_number])
            record["content"] = re.sub(r"^\d+\.", f"{number}.", record["content"])
            record["content"] = nature_reference(record["content"])
            facts = reference_facts(reference_records[source_number]["content"])
            refs[scope].append({"number": number, "source_number": source_number, "text": record["content"],
                                "facts": facts, "facts_conserved": reference_facts(record["content"]) == facts})
            if scope == "main":
                main.append(record)
    for section in ("Acknowledgments", "Funding", "Author contributions", "Competing interests"):
        for record in ms:
            if record["section"] != section:
                continue
            if section == "Funding" and record["kind"] == "h2":
                continue
            rendered = render(record)
            if section == "Acknowledgments" and record["kind"] == "h2":
                rendered["content"] = "Acknowledgements"
            if section == "Funding" and record["kind"] == "p":
                rendered["content"] = "**Funding.** " + rendered["content"]
            main.append(rendered)
    main.append(_generated("h2", "Data availability", "data-availability-heading"))
    for record in availability:
        rendered = render(record)
        rendered["content"] = rendered["content"].replace("The data and code are available", "The data are available")
        main.append(rendered)
    main.extend((_generated("h2", "Code availability", "code-availability-heading"),
                 _generated("p", f"The code is available on Zenodo (DOI: 10.5281/zenodo.23588020) <sup>{main_map[34]}</sup>.", "code-availability")))
    main.append(_generated("h2", "Figure legends", "main-legends-heading"))
    main.extend(render(record) for record in legends)

    front_end = next(index for index, record in enumerate(si) if record["content"] == "**This file includes:**")
    supplementary = [deepcopy(record) for record in si[:front_end]]
    supplementary[0]["content"] = "Supplementary Information for"
    front = ["**This file includes:**", "Supplementary Notes 1\u201320", "Supplementary Figs. 1 to 16",
             "Supplementary Tables 1 to 21", "Supplementary References", "**Contents**"]
    front.extend(record["content"] for record in notes if record["kind"] == "h2")
    front.append("**Figures**")
    front.extend(format_labels(figure_records[number][1]["content"]).split("**", 2)[1].rstrip(".") for number in range(11, 27))
    front.append("**Tables**")
    front.extend(format_labels(record["content"]).split("**", 2)[1].rstrip(".")
                 for record in notes if record["kind"] == "p" and re.match(r"^\*\*Table S\d+ \|", record["content"]))
    supplementary.extend(_generated("p", text, f"si-front-{index}") for index, text in enumerate(front))
    supplementary.extend(render(record) for record in notes)
    supplementary.append(_generated("h2", "References", "si-references-heading"))
    for item in refs["supplementary"]:
        record = deepcopy(reference_records[item["source_number"]])
        record["content"] = item["text"]
        supplementary.append(record)

    extended_data, ed_records = [], []
    for number in range(1, 11):
        image_record, caption_record = figure_records[number]
        image_path = str((root / _SOURCE_SI).parent.joinpath(image_record["content"][1]).resolve())
        image_view, caption_view = render(image_record), render(caption_record)
        image_view["content"] = (f"Extended Data Fig. {number}", image_path)
        extended_data.append({"number": number, "image_path": image_path, "caption": caption_view["content"]})
        ed_records.extend((image_view, caption_view))
    outputs = {"manuscript": main, "supplementary": supplementary, "extended_data": ed_records}
    required = {record["id"] for record in ms if record["kind"] not in {"h1", "h2"}
                and record["section"] not in {"One-sentence summary", "Supplementary Materials"}}
    required.update(record["id"] for record in si if record["source_index"] >= next(
        entry["source_index"] for entry in si if entry["content"] == "Materials and Methods" and entry["kind"] == "h2")
                    and not (record["kind"] == "h2" and record["content"] == "Materials and Methods"))
    manifest = _manifest(root, (_SOURCE_MS, _SOURCE_SI), outputs, required, original)
    return _view(journal, outputs, extended_data, refs, manifest)


def _view(journal, outputs, extended_data, references, manifest):
    view = {"journal": journal,
            "manuscript": [(record["kind"], record["content"]) for record in outputs["manuscript"]],
            "supplementary": [(record["kind"], record["content"]) for record in outputs["supplementary"]],
            "extended_data": extended_data, "references": references, "content_manifest": manifest}
    view["main_legends"] = [content for kind, content in view["manuscript"] if kind == "p" and _MAIN_CAPTION.match(content)]
    view["verification"] = verify_documents(view)
    if view["verification"]["status"] != "PASS":
        raise ValueError(f"Journal view changes scientific content: {view['verification']}")
    return view
