class TextRegion:
    """Rectangle-compatible OCR result that also retains the original polygon."""
    def __new__(cls, xmin, xmax, ymin, ymax, polygon):
        return super().__new__(cls)

    def __init__(self, xmin, xmax, ymin, ymax, polygon):
        self.values = (xmin, xmax, ymin, ymax)
        self.polygon = polygon

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __eq__(self, other):
        return tuple(self.values) == tuple(other)

    def __hash__(self):
        return hash(self.values)


def get_coordinates(dt_box):
    """
    从返回的检测框中获取坐标
    :param dt_box 检测框返回结果
    :return list 坐标点列表
    """
    coordinate_list = list()
    if isinstance(dt_box, list):
        for i in dt_box:
            i = list(i)
            (x1, y1) = int(i[0][0]), int(i[0][1])
            (x2, y2) = int(i[1][0]), int(i[1][1])
            (x3, y3) = int(i[2][0]), int(i[2][1])
            (x4, y4) = int(i[3][0]), int(i[3][1])
            xmin = min(x1, x2, x3, x4)
            xmax = max(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            ymax = max(y1, y2, y3, y4)
            if xmax > xmin and ymax > ymin:
                coordinate_list.append(TextRegion(xmin, xmax, ymin, ymax,
                                                  [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))
    return coordinate_list


def get_polygons(dt_box):
    """Return valid OCR polygons as integer point lists."""
    polygons = []
    if not isinstance(dt_box, list):
        return polygons
    for box in dt_box:
        points = [(int(point[0]), int(point[1])) for point in list(box)]
        if len(points) == 4:
            polygons.append(points)
    return polygons
