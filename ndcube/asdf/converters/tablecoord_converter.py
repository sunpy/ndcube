from asdf.extension import Converter


class TimeTableCoordConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/TimeTableCoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.TimeTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import TimeTableCoordinate

        table = node.get("table")
        names = node.get("names")
        physical_types = node.get("physical_types")
        reference_time = node.get("reference_time")
        timetablecoordinate = TimeTableCoordinate(
            table, names=names, physical_types=physical_types, reference_time=reference_time)

        return timetablecoordinate

    def to_yaml_tree(self, timetablecoordinate, tag, ctx):
        node = {}
        node["table"] = timetablecoordinate.table
        node["names"] = timetablecoordinate.names
        node["mesh"] = timetablecoordinate.mesh
        if timetablecoordinate.physical_types is not None:
            node["physical_types"] = timetablecoordinate.physical_types
        node["reference_time"] = timetablecoordinate.reference_time

        return node


class QuantityTableCoordinateConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/QuantityTableCoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.QuantityTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import QuantityTableCoordinate

        unit = node.get("unit")
        table = node.get("table")
        names = node.get("names")
        mesh = node.get("mesh")
        physical_types = node.get("physical_types")
        quantitytablecoordinate = QuantityTableCoordinate(*table,
                                                          names=names, physical_types=physical_types)
        quantitytablecoordinate.unit = unit
        quantitytablecoordinate.mesh = mesh
        return quantitytablecoordinate

    def to_yaml_tree(self, quantitytablecoordinate, tag, ctx):
        node = {}
        node["unit"] = quantitytablecoordinate.unit
        node["table"] = quantitytablecoordinate.table
        node["names"] = quantitytablecoordinate.names
        node["mesh"] = quantitytablecoordinate.mesh
        if quantitytablecoordinate.physical_types is not None:
            node["physical_types"] = quantitytablecoordinate.physical_types

        return node


class SkyCoordTableCoordinateConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/table_coord/SkyCoordTableCoordinate-*"]
    types = ["ndcube.extra_coords.table_coord.SkyCoordTableCoordinate"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.table_coord import SkyCoordTableCoordinate

        table = node.get("table")
        names = node.get("names")
        mesh = node.get("mesh")
        physical_types = node.get("physical_types")
        skycoordinatetablecoordinate = SkyCoordTableCoordinate(table, mesh=mesh,
                                                               names=names, physical_types=physical_types)
        return skycoordinatetablecoordinate

    def to_yaml_tree(self, skycoordinatetablecoordinate, tag, ctx):
        node = {}
        node["table"] = skycoordinatetablecoordinate.table
        node["names"] = skycoordinatetablecoordinate.names
        node["mesh"] = skycoordinatetablecoordinate.mesh
        if skycoordinatetablecoordinate.physical_types is not None:
            node["physical_types"] = skycoordinatetablecoordinate.physical_types

        return node
