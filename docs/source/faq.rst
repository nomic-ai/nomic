Mapping FAQ
------------
Frequently asked questions about Atlas maps.

Mapping Latency
###############

Map creation latency once Nomic has received your embeddings.

.. csv-table:: Map Creation Latency
   :file: indexing_latency.csv
   :widths: 1, 1
   :header-rows: 1


Who can see my maps?
####################
When you create a map, you can toggle it as private or public. Private maps are only
accessible by authenticated individuals in your Nomic organization. Public maps are accessible by anyone with a link.

Making maps under an organization
#################################
If you are added to a Nomic organization by someone (such as your employer), you can create projects under them
by specifying an `organization_name` in the `map_embedding` method of the AtlasClient. By default, projects are
made under your own account.