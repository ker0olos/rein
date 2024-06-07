import re

import numpy as np

COMMANDS = set("MmZzLlHhVvCcSsQqTtAa")
UPPERCASE = set("MZLHVCSQTA")

COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def compare_bbs(a, b):
    new_bbox = [0, 0, 0, 0]

    new_bbox[0] = min(a[0], b[0])
    new_bbox[1] = max(a[1], b[1])
    new_bbox[2] = min(a[2], b[2])
    new_bbox[3] = max(a[3], b[3])

    return new_bbox


def _normalize(pt, bbox):
    xmin, xmax, ymin, ymax = bbox
    return np.array([(pt[0] - xmin) / (xmax - xmin), (pt[1] - ymin) / (ymax - ymin)])


def _denormalize(pt, bbox):
    xmin, xmax, ymin, ymax = bbox
    return np.array([pt[0] * (xmax - xmin) + xmin, pt[1] * (ymax - ymin) + ymin])


def _bezier_point(control_points, t):
    while len(control_points) > 1:
        p1 = control_points[:-1]
        p2 = control_points[1:]

        control_points = (1 - t) * p1 + t * p2

    return control_points[0]


class Line(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self._bbox = None

        self.d = f"L {self.end[0]},{self.end[1]}"

    def __repr__(self):
        return f"Line(start={self.start}, end={self.end})"

    def __getitem__(self, i):
        match i:
            case 0:
                return self.start
            case 1:
                return self.end

    def __len__(self):
        return 2

    def bbox(self):
        if self._bbox is None:
            self._bbox = (
                min(self.start[0], self.end[0]),
                max(self.start[0], self.end[0]),
                min(self.start[1], self.end[1]),
                max(self.start[1], self.end[1]),
            )

        return self._bbox


class QuadraticBezier(object):
    def __init__(self, start, control, end):
        self.start = start
        self.end = end
        self.control = control

        self._bbox = None

        self.d = f"Q {self.control[0]},{self.control[1]} {self.end[0]},{self.end[1]}"

    def __repr__(self):
        return f"QuadraticBezier(start={self.start}, control={self.control}, end={self.end})"

    def __getitem__(self, i):
        match i:
            case 0:
                return self.start
            case 1:
                return self.control
            case 2:
                return self.end

    def __len__(self):
        return 3

    def bbox(self):
        if self._bbox is None:
            control_points = np.array((self.start, self.control, self.end))

            result = np.empty((3, 2))

            result[0] = self.start
            result[1] = _bezier_point(control_points, 0.5)
            result[2] = self.end

            xs, ys = result.T

            self._bbox = min(xs), max(xs), min(ys), max(ys)

        return self._bbox


class CubicBezier(object):
    def __init__(self, start, control1, control2, end):
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end

        self._bbox = None

        self.d = f"C {self.control1[0]},{self.control1[1]} {self.control2[0]},{self.control2[1]} {self.end[0]},{self.end[1]}"

    def __repr__(self):
        return f"CubicBezier(start={self.start}, control1={self.control1}, control2={self.control2}, end={self.end})"

    def __getitem__(self, i):
        match i:
            case 0:
                return self.start
            case 1:
                return self.control1
            case 2:
                return self.control2
            case 3:
                return self.end

    def __len__(self):
        return 4

    def bbox(self):
        if self._bbox is None:
            control_points = np.array(
                (self.start, self.control1, self.control2, self.end)
            )

            result = np.empty((4, 2))

            result[0] = self.start
            result[1] = _bezier_point(control_points, 0.3333333333333333)
            result[2] = _bezier_point(control_points, 0.6666666666666666)
            result[3] = self.end

            xs, ys = result.T

            self._bbox = min(xs), max(xs), min(ys), max(ys)

        return self._bbox


class Arc(object):
    """forked of https://github.com/mathandy/svgpathtools/blob/19df25b99b405ec4fc7616b58384eca7879b6fd4/svgpathtools/path.py#L1352"""

    def __init__(self, start, radius, rotation, large_arc, sweep, end):
        self.start = start
        self.radius = np.array([abs(radius[0]), abs(radius[1])])
        self.rotation = rotation
        self.large_arc = bool(large_arc)
        self.sweep = bool(sweep)
        self.end = end

        self.phi = np.radians(self.rotation)
        self.rot_matrix = np.exp(1j * self.phi)

        self._bbox = None

        self._parameterize()

        self.d = f"A {self.radius[0]},{self.radius[1]} {self.rotation} {int(self.large_arc):d},{int(self.sweep):d} {self.end[0]},{self.end[1]}"

    def __repr__(self):
        return f"Arc(start={self.start}, radius={self.radius}, rotation={self.rotation}, large_arc={self.large_arc}, sweep={self.sweep}, end={self.end})"

    def __getitem__(self, i):
        match i:
            case 0:
                return self.start
            case 1:
                return self.end
            case -1:
                return self.radius

    def __len__(self):
        return 1

    def point(self, t):
        angle = (self.theta + t * self.delta) * np.pi / 180

        cosphi = self.rot_matrix.real
        sinphi = self.rot_matrix.imag

        rx, ry = self.radius

        x = rx * cosphi * np.cos(angle) - ry * sinphi * np.sin(angle) + self.center.real
        y = rx * sinphi * np.cos(angle) + ry * cosphi * np.sin(angle) + self.center.imag

        return x, y

    def _parameterize(self):
        rx, ry = self.radius

        rx_sqd = rx * rx
        ry_sqd = ry * ry

        zp1 = (1 / self.rot_matrix) * (self.start - self.end) / 2

        x1p, y1p = zp1

        x1p_sqd = x1p * x1p
        y1p_sqd = y1p * y1p

        radius_check = (x1p_sqd / rx_sqd) + (y1p_sqd / ry_sqd)

        if radius_check > 1:
            rx *= np.sqrt(radius_check)
            ry *= np.sqrt(radius_check)
            self.radius = np.array([rx, ry])
            rx_sqd, ry_sqd = rx * rx, ry * ry

        tmp = rx_sqd * y1p_sqd + ry_sqd * x1p_sqd
        radicand = (rx_sqd * ry_sqd - tmp) / tmp
        radical = 0 if np.isclose(radicand, 0) else np.sqrt(radicand)

        if self.large_arc == self.sweep:
            cp = -radical * np.array([rx * y1p / ry, ry * x1p / rx])
        else:
            cp = radical * np.array([rx * y1p / ry, ry * x1p / rx])

        self.center = (
            self.rot_matrix * complex(*cp)
            + (complex(*self.start) + complex(*self.end)) / 2
        )

        u1 = (x1p - cp[0]) / rx + 1j * (y1p - cp[1]) / ry
        u2 = (-x1p - cp[0]) / rx + 1j * (-y1p - cp[1]) / ry

        u1 = np.clip(u1.real, -1, 1) + 1j * np.clip(u1.imag, -1, 1)
        u2 = np.clip(u2.real, -1, 1) + 1j * np.clip(u2.imag, -1, 1)

        if u1.imag > 0:
            self.theta = np.degrees(np.arccos(u1.real))
        elif u1.imag < 0:
            self.theta = -np.degrees(np.arccos(u1.real))
        else:
            if u1.real > 0:
                self.theta = 0
            else:
                self.theta = 180

        det_uv = u1.real * u2.imag - u1.imag * u2.real

        acosand = u1.real * u2.real + u1.imag * u2.imag
        acosand = np.clip(acosand.real, -1, 1) + np.clip(acosand.imag, -1, 1)

        if det_uv > 0:
            self.delta = np.degrees(np.arccos(acosand))
        elif det_uv < 0:
            self.delta = -np.degrees(np.arccos(acosand))
        else:
            if u1.real * u2.real + u1.imag * u2.imag > 0:
                self.delta = 0
            else:
                self.delta = 180

        if not self.sweep and self.delta >= 0:
            self.delta -= 360
        elif self.large_arc and self.delta <= 0:
            self.delta += 360

    def bbox(self):
        if self._bbox is None:
            if np.cos(self.phi) == 0:
                atan_x = np.pi / 2
                atan_y = 0
            elif np.sin(self.phi) == 0:
                atan_x = 0
                atan_y = np.pi / 2
            else:
                rx, ry = self.radius[0], self.radius[1]
                atan_x = np.arctan(-(ry / rx) * np.tan(self.phi))
                atan_y = np.arctan((ry / rx) / np.tan(self.phi))

            def angle_inv(ang, k):
                return (
                    (ang + np.pi * k) * (360 / (2 * np.pi)) - self.theta
                ) / self.delta

            xtrema = [self.start[0], self.end[0]]
            ytrema = [self.start[1], self.end[1]]

            for k in range(-4, 5):
                tx = angle_inv(atan_x, k)
                ty = angle_inv(atan_y, k)

                if 0 <= tx <= 1:
                    xtrema.append(self.point(tx)[0])
                if 0 <= ty <= 1:
                    ytrema.append(self.point(ty)[1])

            self._bbox = min(xtrema), max(xtrema), min(ytrema), max(ytrema)

        return self._bbox


class Path:
    def __init__(self, *segments):
        self._modified_segments = self._segments = []

        if len(segments) > 0:
            self._parse_path(
                segments[0],
                segments[1] if len(segments) >= 2 else np.array([0, 0], "float64"),
            )

        self._bbox = None

    def __repr__(self):
        return "Path({})".format(
            ",\n     ".join(repr(x) for x in self._modified_segments)
        )

    def __len__(self):
        return len(self._modified_segments)

    def d(self):
        parts = []

        current_pos = None

        for segment in self._modified_segments:
            if not np.array_equal(current_pos, segment.start):
                parts.append(f"M {segment.start[0]},{segment.start[1]}")

            parts.append(segment.d)

            current_pos = segment.end

        return " ".join(parts)

    def bbox(self):
        if self._bbox is None:
            bbs = np.array([s.bbox() for s in self._modified_segments])
            xmins, xmaxs, ymins, ymaxs = bbs.T
            self._bbox = np.array(
                [np.min(xmins), np.max(xmaxs), np.min(ymins), np.max(ymaxs)]
            )

        return self._bbox

    def map(self, func, normalize=True):
        new_bbox = None

        self._modified_segments = []

        def _func(segment, bbox, i):
            match len(segment):
                case 2:
                    return Line(
                        start=np.array(func(segment[0], bbox, (i, 0))),
                        end=np.array(func(segment[1], bbox, (i, 1))),
                    )
                case 3:
                    return QuadraticBezier(
                        start=np.array(func(segment[0], bbox, (i, 0))),
                        control=np.array(func(segment[1], bbox, (i, 1))),
                        end=np.array(func(segment[2], bbox, (i, 2))),
                    )
                case 4:
                    return CubicBezier(
                        start=np.array(func(segment[0], bbox, (i, 0))),
                        control1=np.array(func(segment[1], bbox, (i, 1))),
                        control2=np.array(func(segment[2], bbox, (i, 2))),
                        end=np.array(func(segment[3], bbox, (i, 3))),
                    )
                case 1:
                    return Arc(
                        start=np.array(func(segment[0], bbox, (i, 0))),
                        radius=np.array(func(segment[-1], bbox, (i, -1))),
                        rotation=segment.rotation,
                        large_arc=segment.large_arc,
                        sweep=segment.sweep,
                        end=np.array(func(segment[1], bbox, (i, 1))),
                    )

        if normalize:
            _f = func
            func = lambda pt, bbox, i: _denormalize(_f(_normalize(pt, bbox)), bbox)

        for i, seg in enumerate(self._segments):
            # calls the func on each value in the segment
            # supplements the path bbox to the func if needed
            seg = _func(seg, self._bbox, i)

            # get the new segment bbox
            # after any modification is done
            bbox = seg.bbox()

            # this is the path's bbox
            # it's updated after each modification is done
            new_bbox = list(bbox) if new_bbox is None else compare_bbs(new_bbox, bbox)

            self._modified_segments.append(seg)

        return new_bbox

    def _tokenize_path(self, pathdef):
        for x in COMMAND_RE.split(pathdef):
            if x in COMMANDS:
                yield x
            for token in FLOAT_RE.findall(x):
                yield token

    def _parse_path(self, pathdef, current_pos):
        elements = list(self._tokenize_path(pathdef))[::-1]

        start_pos = None
        command = None

        while elements:
            if elements[-1] in COMMANDS:
                last_command = command
                command = elements.pop()
                absolute = command in UPPERCASE
                command = command.upper()
            else:
                if command is None:
                    raise ValueError(
                        f"Unallowed implicit command in {pathdef}, position {len(pathdef.split()) - len(elements)}"
                    )
                last_command = command

            if command == "M":
                pos = np.array([float(elements.pop()), float(elements.pop())])

                if absolute:
                    current_pos = pos
                else:
                    current_pos += pos

                start_pos = current_pos

                command = "L"

            elif command == "Z":
                if not np.array_equal(current_pos, start_pos):
                    self._segments.append(Line(current_pos, start_pos))

                command = None
                current_pos = start_pos

            elif command == "L":
                pos = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    pos += current_pos

                self._segments.append(Line(current_pos, pos))
                current_pos = pos

            elif command == "H":
                pos = np.array([float(elements.pop()), float(current_pos[1])])

                if not absolute:
                    pos[0] += current_pos[0]

                self._segments.append(Line(current_pos, pos))
                current_pos = pos

            elif command == "V":
                pos = np.array([float(current_pos[0]), float(elements.pop())])

                if not absolute:
                    pos[1] += current_pos[1]

                self._segments.append(Line(current_pos, pos))
                current_pos = pos

            elif command == "C":
                control1 = np.array([float(elements.pop()), float(elements.pop())])
                control2 = np.array([float(elements.pop()), float(elements.pop())])
                end = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    control1 += current_pos
                    control2 += current_pos
                    end += current_pos

                self._segments.append(CubicBezier(current_pos, control1, control2, end))
                current_pos = end

            elif command == "S":
                if last_command not in "CS":
                    control1 = current_pos
                else:
                    control1 = current_pos + current_pos - self._segments[-1].control2

                control2 = np.array([float(elements.pop()), float(elements.pop())])
                end = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    control2 += current_pos
                    end += current_pos

                self._segments.append(CubicBezier(current_pos, control1, control2, end))
                current_pos = end

            elif command == "Q":
                control = np.array([float(elements.pop()), float(elements.pop())])
                end = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    control += current_pos
                    end += current_pos

                self._segments.append(QuadraticBezier(current_pos, control, end))
                current_pos = end

            elif command == "T":
                if last_command not in "QT":
                    control = current_pos
                else:
                    control = current_pos + current_pos - self._segments[-1].control

                end = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    end += current_pos

                self._segments.append(QuadraticBezier(current_pos, control, end))
                current_pos = end

            elif command == "A":
                radius = np.array([float(elements.pop()), float(elements.pop())])

                rotation = float(elements.pop())
                arc = float(elements.pop())
                sweep = float(elements.pop())

                end = np.array([float(elements.pop()), float(elements.pop())])

                if not absolute:
                    end += current_pos

                if radius[0] == 0 or radius[1] == 0:
                    self._segments.append(Line(current_pos, end))
                else:
                    self._segments.append(
                        Arc(
                            current_pos,
                            radius,
                            rotation,
                            arc,
                            sweep,
                            end,
                        )
                    )

                current_pos = end
