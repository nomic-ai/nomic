from .atlas import atlas, AtlasClass, ATLAS_DEFAULT_ID_FIELD
from typing import Optional, Union
from loguru import logger
from pyarrow import compute as pc
import pyarrow as pa
import requests
import nomic

class AtlasIndex(AtlasClass):
  """
  An AtlasIndex represents a single view of an Atlas Project at a point in time.

  An AtlasIndex typically contains one or more *projections* which are 2d representations of
  the points in the index that you can browse online.
  """

  pass


class AtlasProject(AtlasClass):
    
  def __init__(self,
    name : str = None,
    description : Optional[str] = None,
    unique_id_field : Optional[str] = None,
    modality : Optional[str] = None,
    organization_name: Optional[str] = None,
    is_public: bool = True,
    project_id = None
    ):
    """
    Creates or loads an Atlas project.
    Atlas projects store data (text, embeddings, etc) that you can organize by building indices.
    If the organization already contains a project with this name, it will be returned instead.

    **Parameters:**

    * **project_name** - The name of the project.
    * **description** - A description for the project.
    * **unique_id_field** - The field that uniquely identifies each datum. If a datum does not contain this field, it will be added and assigned a random unique ID.
    * **modality** - The data modality of this project. Currently, Atlas supports either `text` or `embedding` modality projects.
    * **organization_name** - The name of the organization to create this project under. You must be a member of the organization with appropriate permissions. If not specified, defaults to your user account's default organization.
    * **is_public** - Should this project be publicly accessible for viewing (read only). If False, only members of your Nomic organization can view.
    * **reset_project_if_exists** - If the requested project exists in your organization, will delete it and re-create it.
    * **project_id** - An alternative way to retrieve a project is by passing the project_id directly. This only works if a project exissts.
    **Returns:** project_id on success.

    """
    assert name is not None or project_id is not None, "You must pass a name or project_id"

    super().__init__()

    if project_id is not None:
      self.meta = self._get_project_by_id(project_id) 
      self.name = self.meta['project_name']
      return

    self.name = name
    try:
      self.meta = self.get_project(self.name)   
    except Exception as e:
      if "Could not find project" in str(e):
        assert description is not None, "You must provide a description when creating a new project."
        assert modality is not None, "You must provide a modality when creating a new project."        
        logger.info(f"Creating project: {self.name}")
        if unique_id_field is None:
          unique_id_field = ATLAS_DEFAULT_ID_FIELD
        atlas.create_project(
          self.name,
          description=description,
          unique_id_field=unique_id_field,
          modality=modality,
          organization_name=organization_name,
          is_public=is_public,
          reset_project_if_exists=False)
        self.meta = self.get_project(self.name)    

  def project_info(self):
    response = requests.get(
        self.atlas_api_path+ f"/v1/project/{self.id}",
        headers=self.header,
    )
    return response.json()

  @property
  def indices(self):
    return self.project_info()['atlas_indices']
    
  @property
  def projections(self):
    vs = []
    for index in self.indices:      
      for projection in index['projections']:
        vs.append(nomic.AtlasProjection(self, projection['id']))
    return vs

  @property
  def header(self):
    return self.header

  @property
  def id(self):
    return self.meta['id']


  @property
  
  def __repr__(self):
    m = self.meta
    return f"Nomic project: <{m}>"

  def upload_embeddings(self, table : pa.Table, embedding_column : str = '_embedding'):
    """
    Uploads a single Arrow table to Atlas.
    """ 
    dimensions = table[embedding_column].type.list_size
    embs = pc.list_flatten(table[embedding_column]).to_numpy().reshape(-1, dimensions)
    self.atlas_client.add_embeddings(
      project_id = self.id,
      embeddings = embs,
      data = table.drop([embedding_column]).to_pylist(),
      shard_size = 1500)