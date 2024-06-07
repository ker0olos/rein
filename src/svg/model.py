from collections import namedtuple
from xml.dom.expatbuilder import parseString

import numpy as np

from src.svg.path import Path, compare_bbs
from src.svg.utils import (
    attrs_to_string,
    define_groups,
    dom_to_dict,
    ellipse_to_pathd,
    get_clip_paths,
    lines_to_pathd,
    polygon_to_pathd,
    polyline_to_pathd,
    purge,
    rect_to_pathd,
)

Group = namedtuple("Group", ["key", "attributes", "compiled", "children"])


class Model:
    def __init__(self, xml_string: str):
        if xml_string is None:
            return

        doc = purge(define_groups(parseString(xml_string)))

        metadata, defs = dom_to_dict(
            doc.getElementsByTagName("svg")[0]
        ), get_clip_paths(doc)

        paths = [dom_to_dict(el) for el in doc.getElementsByTagName("path")]
        groups = [dom_to_dict(el) for el in doc.getElementsByTagName("g")]

        lines = [dom_to_dict(el) for el in doc.getElementsByTagName("line")]
        polylines = [dom_to_dict(el) for el in doc.getElementsByTagName("polyline")]
        polygons = [dom_to_dict(el) for el in doc.getElementsByTagName("polygon")]
        circles = [dom_to_dict(el) for el in doc.getElementsByTagName("circle")]
        ellipses = [dom_to_dict(el) for el in doc.getElementsByTagName("ellipse")]
        rectangles = [dom_to_dict(el) for el in doc.getElementsByTagName("rect")]

        self.zoom_level = float(metadata.get("rein:canvas-zoom", "1"))
        self.sensitivity = float(metadata.get("rein:pose-sensitivity", "0.8"))
        self.tracking = metadata.get("rein:tracking", "face")

        self.face_rotation = 0
        self.face_origin = (0, 0)

        self.arms_rotation = [0, 0]
        self.arms_origin = [(), ()]

        self.eyebrow_diff = [0, 0]

        self.defs = defs
        self.groups = groups

        self.attributes = (
            paths + lines + polylines + polygons + circles + ellipses + rectangles
        )

        self.compiled_attrs = []

        self.paths = [
            Path(d)
            for d in (
                [el["d"] for el in paths]
                + [lines_to_pathd(li) for li in lines]
                + [polyline_to_pathd(pl) for pl in polylines]
                + [polygon_to_pathd(pg) for pg in polygons]
                + [ellipse_to_pathd(c) for c in circles]
                + [ellipse_to_pathd(e) for e in ellipses]
                + [rect_to_pathd(r) for r in rectangles]
            )
        ]

        # allow us to find paths and groups
        # by using their id
        self.dictionary = {}

        # define groups in the dictionary
        for i, attributes in enumerate(self.groups):
            id = attributes["id"]

            if id in self.dictionary:
                print(
                    f'WARNING:DUPLICATED_IDS: two or more groups share the same id: "{id}"'
                )

            self.dictionary[id] = Group(i, attributes, attrs_to_string(attributes), [])

        for i, attributes in enumerate(self.attributes):
            parent = None

            if "parent-id" in attributes:
                parent = self.dictionary[attributes["parent-id"]]

            if "id" in attributes:
                id = attributes["id"]

                if id in self.dictionary:
                    print(
                        f'WARNING:DUPLICATED_IDS: two or more elements share the same id: "{id}"'
                    )

                self.dictionary[id] = i

            # add the child to the parent group
            if parent is not None:
                parent.children.append(i)

            # saves a compiled string of the attributes
            # to avoid an unnecessary loop later
            self.compiled_attrs.append(attrs_to_string(attributes))

        bbs = np.array([p.bbox() for p in self.paths])

        xmins, xmaxs, ymins, ymaxs = bbs.T
        self.virgin_bbox = np.min(xmins), np.max(xmaxs), np.min(ymins), np.max(ymaxs)
        self.bbox = list(self.virgin_bbox)

        if self.__exists__("face"):
            self.face_virgin_bbox = self.__find_bbox_by_id__("face")

        if self.__exists__("left-eye") and self.__exists__("left-eyebrow"):
            left_eye_bbox, left_eyebrow_bbox = (
                self.__find_bbox_by_id__("left-eye"),
                self.__find_bbox_by_id__("left-eyebrow"),
            )
            self.eyebrow_diff[0] = left_eye_bbox[2] - left_eyebrow_bbox[3]

        if self.__exists__("right-eye") and self.__exists__("right-eyebrow"):
            right_eye_bbox, right_eyebrow_bbox = (
                self.__find_bbox_by_id__("right-eye"),
                self.__find_bbox_by_id__("right-eyebrow"),
            )
            self.eyebrow_diff[1] = right_eye_bbox[2] - right_eyebrow_bbox[3]

        doc.unlink()

    def __repr__(self):
        return "Model({})".format(",\n     ".join(repr(x) for x in self.paths))

    def __len__(self):
        return len(self.paths)

    def __exists__(self, id):
        return id in self.dictionary

    def __find_attrs_by_id__(self, id, key, default):
        if self.__exists__(id):
            item = self.dictionary[id]

            # if group return the attrs of the group instead
            if isinstance(item, Group):
                return self.groups[item.key].get(key, default)
            else:
                return self.attributes[item].get(key, default)
        else:
            return default

    def __find_bbox_by_id__(self, id):
        if self.__exists__(id):
            item = self.dictionary[id]

            # process all the children of the group
            if isinstance(item, Group):
                bbs = np.array([self.paths[key].bbox() for key in item.children])
            else:
                bbs = self.paths[item].bbox()

        xmins, xmaxs, ymins, ymaxs = bbs.T
        bbox = np.min(xmins), np.max(xmaxs), np.min(ymins), np.max(ymaxs)
        return bbox

    def __map__(self, func, id, normalize=True):
        # most certainly won't be called on a group
        # there's no need to check
        path = self.paths[self.dictionary[id]]

        new_bbox = path.map(func, normalize)

        # update the model's bbox
        self.bbox = compare_bbs(self.bbox, new_bbox)

        return new_bbox

    def __apply_eyes__(self, eyelids, eyebrows, irises):
        def _map_eye(id, height):
            xerr = float(self.__find_attrs_by_id__(id, "rein:xerr", "0.03"))
            yerr = float(self.__find_attrs_by_id__(id, "rein:yerr", "0.015"))

            scale = float(self.__find_attrs_by_id__(id, "rein:scale", "1"))

            closed = 0.020
            regular = 0.052

            if scale > 0:
                normal = max(((height - closed) / (regular - closed), 0)) * scale * 0.5
                normal = 0.5 - normal
            else:
                normal = 0

            def func(pt):
                x, y = pt

                if y > 0.5 and y - yerr - normal > 0.5:
                    y = y - normal
                elif y + yerr + normal < 0.5:
                    y = y + normal
                else:
                    y = 0.5

                if x - xerr < 0:
                    x = 0
                elif x + xerr > 1:
                    x = 1

                return x, y

            return self.__map__(func, id)

        def _map_brow(id, eye_bbox, real_diff, model_diff):
            if not self.__exists__(id):
                return

            scale = float(self.__find_attrs_by_id__(id, "rein:scale", "1"))

            diff = (
                (real_diff - 0.12)
                * (self.face_virgin_bbox[3] - self.face_virgin_bbox[2])
                * scale
            )

            def func(pt, eyebrow_bbox, _):
                return (
                    pt[0],
                    eye_bbox[2]
                    + (pt[1] - eyebrow_bbox[2])
                    - (eyebrow_bbox[3] - eyebrow_bbox[2])
                    - model_diff
                    - diff,
                )

            self.__map__(func, id, normalize=False)

        def _map_iris(id, real_diff):
            if not self.__exists__(id):
                return

            scale = float(self.__find_attrs_by_id__(id, "rein:scale", "1"))

            diff = (
                real_diff
                * [
                    self.face_virgin_bbox[1] - self.face_virgin_bbox[0],
                    self.face_virgin_bbox[3] - self.face_virgin_bbox[2],
                ]
                * scale
            )

            def func(pt, _, __):
                return pt + diff

            self.__map__(func, id, normalize=False)

        if self.__exists__("left-eye"):
            eye_bbox = _map_eye("left-eye", eyelids[0])
            _map_brow("left-eyebrow", eye_bbox, eyebrows[0], self.eyebrow_diff[0])
            _map_iris("left-iris", irises)

        if self.__exists__("right-eye"):
            eye_bbox = _map_eye("right-eye", eyelids[1])
            _map_brow("right-eyebrow", eye_bbox, eyebrows[1], self.eyebrow_diff[1])
            _map_iris("right-iris", irises)

    def __apply_mouth__(self, width, height):
        if not self.__exists__("mouth"):
            return

        xerr = float(self.__find_attrs_by_id__("mouth", "rein:xerr", "0.03"))
        yerr = float(self.__find_attrs_by_id__("mouth", "rein:yerr", "0.015"))

        scale = float(self.__find_attrs_by_id__("mouth", "rein:scale", "1"))

        x_regular = 0.34
        y_regular = 0.062

        if scale > 0:
            x_normal = min((width - x_regular) / x_regular, 0) * scale
            y_normal = max((height - y_regular) / y_regular, 0) * scale
            y_normal = 1 - y_normal
        else:
            x_normal, y_normal = 0, 0

        def func(pt):
            x, y = pt

            if y > 0.5 and y - yerr - y_normal > 0.5:
                y = y - y_normal
            elif y + yerr + y_normal < 0.5:
                y = y + y_normal
            else:
                y = 0.5

            if x > 0.5 and x + xerr + x_normal > 0.5:
                x = x + x_normal
            elif x - xerr - x_normal < 0.5:
                x = x - x_normal
            else:
                x = 0.5

            return x, y

        self.__map__(func, "mouth")

    def tostring(self):
        clippath = (
            lambda id: f'<clipPath id="{id}-clippath"><use href="#{id}"/></clipPath>'
        )

        path = lambda attrs: f"<path {attrs}/>"

        group = lambda attrs: f"<g {attrs}>"

        transformed_group = lambda r, o: (
            f'<g transform="rotate({r})" style="transform-origin:{o[0]}px {o[1]}px">'
            if len(o) == 2
            else "<g>"
        )

        defs = []
        children = []
        opened_groups = []

        def close_group(parent_id):
            if parent_id == "face":
                children.append("</g>")
            children.append("</g>")

        for i, p in enumerate(self.paths):
            attributes = self.attributes[i]

            parent_id = attributes.get("parent-id", None)

            if parent_id is not None and parent_id not in opened_groups:
                # add wrappers for parts that require transformation
                match parent_id:
                    case "face":
                        children.append(
                            transformed_group(self.face_rotation, self.face_origin)
                        )
                    # case "left-arm":
                    #     was_wrapper_group = True
                    #     children.append(
                    #         transformed_group(
                    #             self.arms_rotation[0], self.arms_origin[0]
                    #         )
                    #     )
                    # case "right-arm":
                    #     was_wrapper_group = True
                    #     children.append(
                    #         transformed_group(
                    #             self.arms_rotation[1], self.arms_origin[1]
                    #         )
                    #     )

                children.append(group(self.dictionary[parent_id].compiled))
                opened_groups.append(parent_id)

            attrs = [self.compiled_attrs[i]]

            allow_clippath = self.attributes[i].get("rein:clippath", "on")

            match self.attributes[i].get("id"):
                case "right-eye" | "left-eye":
                    if allow_clippath == "on":
                        defs.append(clippath(self.attributes[i]["id"]))
                case "left-iris":
                    attrs.append('clip-path="url(#left-eye-clippath)"')
                case "right-iris":
                    attrs.append('clip-path="url(#right-eye-clippath)"')

            attrs.append(f'd="{p.d()}"')

            children.append(path(" ".join(attrs)))

            if parent_id is not None and i == self.dictionary[parent_id].children[-1]:
                close_group(parent_id)

        xmin, xmax, ymin, ymax = self.bbox

        width, height = xmax - xmin, ymax - ymin

        cx = round(xmin + (width / 2))
        cy = round(ymin + (height / 2))

        canvas_width, canvas_height = 1280, 720

        canvas_x = (canvas_width * 0.5) - (width * 0.5) - xmin
        canvas_y = (canvas_height * 0.5) - (height * 0.5) - ymin

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="0,0,{canvas_width},{canvas_height}" width="100%">
            <g transform="translate({canvas_x}, {canvas_y}) scale({self.zoom_level})" style="transform-origin: {cx}px {cy}px">
                <defs>
                    {"".join(defs + self.defs)}
                </defs>
                {"".join(children)}
            </g>
        </svg>
        """

        return xml
