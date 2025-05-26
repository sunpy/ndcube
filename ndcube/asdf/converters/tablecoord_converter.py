from asdf.extension import Converter


class TimeTableCoordConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/timetablecoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.TimeTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import TimeTableCoordinate

        names = node.get("names")
        physical_types = node.get("physical_types")
        reference_time = node.get("reference_time")
        return TimeTableCoordinate(
            node["table"],
            names=names,
            physical_types=physical_types,
            reference_time=reference_time,
        )

    def to_yaml_tree(self, timetablecoordinate, tag, ctx):
        node = {}
        node["table"] = timetablecoordinate.table
        if timetablecoordinate.names:
            node["names"] = timetablecoordinate.names
        if timetablecoordinate.physical_types is not None:
            node["physical_types"] = timetablecoordinate.physical_types
        node["reference_time"] = timetablecoordinate.reference_time

        return node


class QuantityTableCoordinateConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/quantitytablecoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.QuantityTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import QuantityTableCoordinate

        names = node.get("names")
        mesh = node.get("mesh")
        physical_types = node.get("physical_types")
        quantitytablecoordinate = QuantityTableCoordinate(*node["table"], names=names, physical_types=physical_types)
        quantitytablecoordinate.unit = node["unit"]
        quantitytablecoordinate.mesh = mesh
        return quantitytablecoordinate

    def to_yaml_tree(self, quantitytablecoordinate, tag, ctx):
        node = {}
        node["unit"] = quantitytablecoordinate.unit
        node["table"] = quantitytablecoordinate.table
        if quantitytablecoordinate.names:
            node["names"] = quantitytablecoordinate.names
        node["mesh"] = quantitytablecoordinate.mesh
        if quantitytablecoordinate.physical_types is not None:
            node["physical_types"] = quantitytablecoordinate.physical_types

        return node


class SkyCoordTableCoordinateConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/skycoordtablecoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.SkyCoordTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import SkyCoordTableCoordinate

        names = node.get("names")
        mesh = node.get("mesh")
        physical_types = node.get("physical_types")
        return SkyCoordTableCoordinate(node["table"], mesh=mesh, names=names, physical_types=physical_types)

    def to_yaml_tree(self, skycoordinatetablecoordinate, tag, ctx):
        node = {}
        node["table"] = skycoordinatetablecoordinate.table
        if skycoordinatetablecoordinate.names:
            node["names"] = skycoordinatetablecoordinate.names
        node["mesh"] = skycoordinatetablecoordinate.mesh
        if skycoordinatetablecoordinate.physical_types is not None:
            node["physical_types"] = skycoordinatetablecoordinate.physical_types

        return node


class MultipleTableCoordinateConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/multipletablecoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.MultipleTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import MultipleTableCoordinate

        mtc = MultipleTableCoordinate(*node["table_coords"])
        mtc._dropped_coords = node["dropped_coords"]
        return mtc

    def to_yaml_tree(self, multipletablecoordinate, tag, ctx):
        node = {}
        node["table_coords"] = multipletablecoordinate._table_coords
        node["dropped_coords"] = multipletablecoordinate._dropped_coords
        return node
