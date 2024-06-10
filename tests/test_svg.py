import xml.etree.ElementTree as ET
from xml.dom import minidom

from src.svg.model import Model

xml_ellipse_string = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg>
    <g id="face" transform="translate(5, 5)">
        <ellipse
            style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0.837346;stroke-linejoin:round"
            cx="24"
            cy="24"
            rx="23.581327"
            ry="23.581326" />
    </g>
</svg>
"""

xml_curve_string = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns:rein="https://github.com/ker0olos/rein">
    <defs>
        <clipPath id="left">
            <use href="#left-2"/>
        </clipPath>
    </defs>
    <path
        rein:scale="0.085"
        rein:clippath="on"
        id="left-eye"
        d="m 82.839512,110.0332 c 0,0 -0.253523,-0.65883 -0.219247,-0.99757 0.03246,-0.32073 0.02704,-0.48562 0.207912,-0.62812 0.237437,-0.0791 -0.0164,-0.0675 0.867893,0.0945 0.182144,-0.0496 0.895673,0.3526 0.864703,0.54765 -0.0059,0.037 -0.05827,0.36559 -0.118657,0.54065 -0.0527,0.15279 -0.120844,0.4527 -0.210738,0.43934 -0.250592,-0.0372 -0.612432,0.003 -0.737249,-0.01 -0.378201,-0.0383 -0.654614,0.0133 -0.654614,0.0133 z"
        style="opacity:1;fill:#ffffff;fill-opacity:0.5;fill-rule:evenodd;stroke:#000000;stroke-width:0.132292;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1;paint-order:normal" />
</svg>
"""


xml_inkscape_string = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">
    <g inkscape:label="Layer 1" id="layer1">
        <ellipse
            id="path-7874922389283097748029598590584979208590"
            inkscape:label="Elliope"
            style="fill:#000000;fill-opacity:1;stroke:#000000;stroke-width:0.837346;stroke-linejoin:round"
            cx="24"
            cy="24"
            rx="23.581327"
            ry="23.581326" />
    </g>
</svg>
"""


def test_parsing(snapshot):
    model = Model(xml_ellipse_string)

    snapshot.assert_match(str(model), "test_parsing.xml")


def test_parsing_2(snapshot):
    model = Model(xml_curve_string)

    snapshot.assert_match(str(model), "test_parsing_2.xml")


def test_bbox(snapshot):
    model = Model(xml_ellipse_string)

    snapshot.assert_match(str(model.bbox), "test_bbox.xml")


def test_bbox_2(snapshot):
    model = Model(xml_curve_string)

    snapshot.assert_match(str(model.bbox), "test_bbox_2.xml")


def test_inkscape_labels():
    model = Model(xml_inkscape_string)

    element = minidom.parseString(model.tostring())

    assert element.getElementsByTagName("g")[1].attributes["id"].value == "Layer 1"
    assert element.getElementsByTagName("path")[0].attributes["id"].value == "Elliope"


def test_tostring(snapshot):
    model = Model(xml_ellipse_string)

    element = ET.XML(model.tostring())

    ET.indent(element)

    snapshot.assert_match(ET.tostring(element, encoding="unicode"), "test_tostring.xml")


def test_tostring_2(snapshot):
    model = Model(xml_curve_string)

    element = ET.XML(model.tostring())

    ET.indent(element)

    snapshot.assert_match(
        ET.tostring(element, encoding="unicode"), "test_tostring_2.xml"
    )
