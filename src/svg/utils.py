# forked from https://github.com/mathandy/svgpathtools
from xml.dom.minidom import Element, Document


def dom_to_dict(element: Element):
    keys = element.attributes.keys()
    values = [val.value for val in element.attributes.values()]
    _dict = dict(zip(keys, values))

    # convert `id` to `inkscape:label` if label is found
    if "inkscape:label" in _dict:
        _dict.update({"id": _dict["inkscape:label"]})

    return _dict


def get_clip_paths(doc: Document):
    defs = []

    for d in doc.getElementsByTagName("defs"):
        for el in d.childNodes:
            if el.nodeType != 3 and el.tagName in ["clipPath"]:
                defs.append(el.toxml())

    return defs


def define_groups(doc: Document):
    for i, g in enumerate(doc.getElementsByTagName("g")):
        # convert `id` to `inkscape:label` if label is found
        if g.hasAttribute("inkscape:label"):
            g.setAttribute("id", g.getAttribute("inkscape:label"))
        elif not g.hasAttribute("id"):
            g.setAttribute("id", f"_rein_group_{i}")

        id = g.getAttribute("id")

        for el in g.childNodes:
            if el.nodeType != 3:
                el.setAttribute("parent-id", id)

    return doc


def purge(doc: Document):
    for g in doc.getElementsByTagName("g"):
        display, style = g.getAttribute("display"), g.getAttribute("style")
        if display == "none" or "display:none" in style:
            g.parentNode.removeChild(g)

    for path in doc.getElementsByTagName("path"):
        d, display, style = (
            path.getAttribute("d"),
            path.getAttribute("display"),
            path.getAttribute("style"),
        )
        if d in ["m0 0", "M 0,0"] or display == "none" or "display:none" in style:
            path.parentNode.removeChild(path)

    return doc


def attrs_to_string(attributes: dict):
    attrs = []
    for key in attributes:
        if key in [
            "id",
            "fill",
            "stroke",
            "stroke-width",
            "stroke-linecap",
            "stroke-linejoin",
            "transform",
            "clip-path",
            "style",
        ]:
            attrs.append(f'{key}="{attributes[key]}"')
    return " ".join(attrs)


def lines_to_pathd(line):
    x0, y0 = line["x1"], line["y1"]
    x1, y1 = line["x2"], line["y2"]

    return f"M {x0} {y0} L {x1} {y1}"


def rect_to_pathd(rect):
    x0, y0 = float(rect.get("x", 0)), float(rect.get("y", 0))

    w, h = float(rect.get("width", 0)), float(rect.get("height", 0))

    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    return f"M{x0} {y0} L {x1} {y1} L {x2} {y2} L {x3} {y3} z" ""


def ellipse_to_pathd(ellipse):
    cx = ellipse.get("cx", 0)
    cy = ellipse.get("cy", 0)

    r = ellipse.get("r", None)
    rx = ellipse.get("rx", None)
    ry = ellipse.get("ry", None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = f"M {cx - rx},{cy}"

    d += f"a{rx},{ry} 0 1,0 {2 * rx},0"
    d += f"a{rx},{ry} 0 1,0 {-2 * rx},0"

    return d


def polygon_to_pathd(polyline):
    return polyline_to_pathd(polyline, True)


def polyline_to_pathd(polyline, is_polygon=False):
    points = polyline

    closed = float(points[0][0]) == float(points[-1][0]) and float(
        points[0][1]
    ) == float(points[-1][1])

    if is_polygon and closed:
        points.append(points[0])

    d = "M" + "L".join(f"{x} {y}" for x, y in points)

    if is_polygon or closed:
        d += "z"

    return d
