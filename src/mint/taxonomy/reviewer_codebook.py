from __future__ import annotations

from collections import defaultdict
from html import escape
import textwrap

GUIDANCE = (
    "Classify the logical form of the obligation separately from its validity. "
    "Read the source discussion and the entire contract. Identify the constrained "
    "operation, arguments, return value, environment, selection, or ordering. "
    "Compare the definitions, inclusion criteria, and exclusions, then choose one "
    "most specific leaf. Branch nodes are organizational and cannot be assigned. "
    "Confirm the current assignment only when it fits. Otherwise choose a different "
    "leaf and explain the boundary. Use Unclassified if no leaf fits, or Uncertain "
    "when the evidence cannot distinguish plausible leaves. Do not create a new "
    "category during annotation. Record taxonomy problems for the meeting."
)


def tree_svg(taxonomy: dict) -> str:
    nodes = taxonomy["nodes"]
    children = defaultdict(list)
    for node in nodes:
        children[node.get("parent_id")].append(node)
    positions = {}
    cursor = 145

    def place(node, depth):
        nonlocal cursor
        descendants = children[node["id"]]
        if descendants:
            ys = [place(child, depth + 1) for child in descendants]
            y = (ys[0] + ys[-1]) / 2
        else:
            y = cursor
            cursor += 100
        positions[node["id"]] = (35 + depth * 365, y)
        return y

    roots = children[None]
    root_ys = [place(node, 1) for node in roots]
    positions["__root__"] = (35, (root_ys[0] + root_ys[-1]) / 2)
    width = max(x for x, _ in positions.values()) + 340
    height = cursor + 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Complete MINT taxonomy tree</title>',
        '<desc id="desc">Every frozen taxonomy branch and leaf. Green indicates an EMSE leaf; blue indicates a MINT extension; gray indicates a branch.</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        '<g font-family="Arial, sans-serif">',
        '<text x="35" y="40" font-size="25" font-weight="700" fill="#192a28">MINT contract-type taxonomy</text>',
        f'<text x="35" y="68" font-size="15" fill="#4c5b59">{escape(taxonomy.get("version", ""))} · all branches and leaves</text>',
        '<text x="35" y="94" font-size="14" fill="#4c5b59">Gray: branch · Green: EMSE leaf · Blue: MINT extension · Assign leaves only</text>',
    ]
    for node in nodes:
        parent = node.get("parent_id") or "__root__"
        px, py = positions[parent]
        x, y = positions[node["id"]]
        mid = px + 330
        parts.append(
            f'<path d="M {px + 310} {py} H {mid} V {y} H {x}" fill="none" stroke="#92a39e" stroke-width="2"/>'
        )
    boxes = [{"id": "__root__", "name": "Contract types"}, *nodes]
    for node in boxes:
        x, y = positions[node["id"]]
        branch = node["id"] == "__root__" or bool(children[node["id"]])
        fill = (
            "#edf0ee"
            if branch
            else (
                "#dcecf7"
                if node.get("taxonomy_origin") == "mint_extension"
                else "#e1efe5"
            )
        )
        parts.append(
            f'<rect x="{x}" y="{y-38}" width="310" height="76" rx="9" fill="{fill}" stroke="#738e82"/>'
        )
        if node["id"] != "__root__":
            parts.append(
                f'<text x="{x+12}" y="{y-16}" font-size="13" font-weight="700" fill="#28564a">{escape(node["id"])} · {"branch" if branch else "leaf"}</text>'
            )
        for i, line in enumerate(textwrap.wrap(node["name"], 36)):
            parts.append(
                f'<text x="{x+12}" y="{y+5+i*18}" font-size="15" fill="#192a28">{escape(line)}</text>'
            )
    parts.append("</g></svg>")
    return "".join(parts)


def codebook_markdown(taxonomy: dict) -> str:
    parents = {node.get("parent_id") for node in taxonomy["nodes"]}
    version_label = "Frozen version" if taxonomy.get("frozen") else "Draft version"
    lines = [
        "# MINT reviewer codebook",
        "",
        f"{version_label}: `{taxonomy['version']}`.",
        "",
        "![Complete taxonomy tree](taxonomy-tree.svg)",
        "",
        "## Assignment procedure",
        "",
        GUIDANCE,
        "",
        "The current automated leaf is visible for verification. Record your agreement, a different leaf, or uncertainty. "
        "This is assignment verification, not a blind taxonomy classification exercise. "
        "Both reviewers work independently before reconciliation.",
        "",
        "Definitions and criteria below reproduce the saved taxonomy snapshot. EMSE seed definitions "
        "remain immutable; MINT extensions are explicitly identified.",
        "",
    ]
    for node in taxonomy["nodes"]:
        kind = "Branch" if node["id"] in parents else "Assignable leaf"
        lines += [
            f"## {node['id']} — {node['name']}",
            "",
            f"{kind}. Parent: `{node.get('parent_id') or 'root'}`. Origin: `{node.get('taxonomy_origin', '')}`.",
            "",
            node.get("definition", ""),
            "",
            "**Include**",
            "",
        ]
        lines += [f"- {item}" for item in node.get("inclusion_criteria", [])] or [
            "- No additional inclusion criteria."
        ]
        lines += ["", "**Exclude / distinguish from**", ""]
        lines += [f"- {item}" for item in node.get("exclusion_criteria", [])] or [
            "- No additional exclusions."
        ]
        lines += ["", "**Citations and source references**", ""]
        lines += [f"- {item}" for item in node.get("citations", [])] or [
            "- See frozen taxonomy provenance."
        ]
        lines += [""]
    return "\n".join(lines)


def codebook_html(taxonomy: dict) -> str:
    parents = {node.get("parent_id") for node in taxonomy["nodes"]}
    version_label = "Frozen version" if taxonomy.get("frozen") else "Draft version"
    sections = []
    for node in taxonomy["nodes"]:

        def items(key):
            return (
                "".join(f"<li>{escape(item)}</li>" for item in node.get(key, []))
                or "<li>No additional criteria.</li>"
            )

        sections.append(
            f'<section id="{escape(node["id"])}"><h2>{escape(node["id"])} — {escape(node["name"])}</h2>'
            f'<p class="meta">{"Branch" if node["id"] in parents else "Assignable leaf"} · Parent: {escape(node.get("parent_id") or "root")} · Origin: {escape(node.get("taxonomy_origin", ""))}</p>'
            f'<p>{escape(node.get("definition", ""))}</p><h3>Include</h3><ul>{items("inclusion_criteria")}</ul>'
            f'<h3>Exclude / distinguish from</h3><ul>{items("exclusion_criteria")}</ul>'
            f'<details><summary>Citations and source references</summary><ul>{items("citations")}</ul></details></section>'
        )
    links = " ".join(
        f'<a href="#{escape(node["id"])}">{escape(node["id"])}</a>'
        for node in taxonomy["nodes"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>MINT reviewer codebook</title><style>body{max-width:1100px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#19362d;background:#fbfaf6}h1,h2,h3{line-height:1.3}section{padding:20px 0;border-top:1px solid #ccd8d1}a{color:#285d85;margin-right:12px}img{width:100%;height:auto;background:white;border:1px solid #ccd8d1}.meta{color:#52665e;font-size:14px}nav{line-height:2.2}</style>"
        f'<h1>MINT reviewer codebook</h1><p>{version_label}: {escape(taxonomy.get("version", ""))} · {len(taxonomy["nodes"])} taxonomy nodes</p>'
        '<p><a href="taxonomy-tree.svg" target="_blank">Open / download full-size tree</a></p>'
        '<img src="taxonomy-tree.svg" alt="Complete taxonomy tree with every branch and leaf">'
        f"<h2>Assignment procedure</h2><p>{escape(GUIDANCE)}</p>"
        "<p>The current automated leaf is visible for verification. Record your agreement, a different leaf, or uncertainty. Review independently before reconciliation.</p>"
        f'<nav aria-label="Taxonomy definitions">{links}</nav>{"".join(sections)}</html>'
    )
