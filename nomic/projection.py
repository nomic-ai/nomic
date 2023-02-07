import io
import json
from pathlib import Path
from typing import Union

import requests
from pyarrow import feather, ipc

from .project import AtlasClass, AtlasProject


class AtlasProjection(AtlasClass):
    def __init__(
        self,
        project: Union[AtlasProject, str],
        id: str,
    ):
        """
        Creates or loads an Atlas projection.
        """
        super().__init__()
        if type(project) == str:
            project = AtlasProject(project)
        self.project = project
        self.meta = None
        for index in project.indices:
            for projection in index['projections']:
                if projection['id'] == id:
                    self.meta = projection
                    self.id = id
                    self.index = index
        if self.meta is None:
            raise ValueError(f"{id} not found in project {project}")

    def _download_feather(self, dest="tiles"):
        """
        Downloads the feather tree.

        """
        dest = Path(dest)
        root = f'https://staging-api-atlas.nomic.ai/v1/project/public/{self.project.id}/index/projection/{self.id}/quadtree/'
        quads = [f'0/0/0']
        while len(quads) > 0:
            quad = quads.pop(0) + ".feather"
            path = dest / quad
            if not path.exists():
                data = requests.get(root + quad)
                readable = io.BytesIO(data.content)
                readable.seek(0)
                tb = feather.read_table(readable)
                path.parent.mkdir(parents=True, exist_ok=True)
                feather.write_feather(tb, path)
            schema = ipc.open_file(path).schema
            kids = schema.metadata.get(b'children')
            children = json.loads(kids)
            quads.extend(children)
