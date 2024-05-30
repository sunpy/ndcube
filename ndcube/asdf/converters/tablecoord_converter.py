from asdf.extension import Converter


class TableCoordConverter(Converter):
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
