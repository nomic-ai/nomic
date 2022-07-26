from nomic import AtlasClient

atlas = AtlasClient()


print(atlas._get_user())

# projects = atlas.get_projects()
#
# print("Available projects:", projects)
# if not projects:
#     exit(0)
#
# project = projects[0]
#
# #Get currently tagged documents
#
# tagged_documents = atlas.get_documents_by_tags(project=project)
#
# if tagged_documents:
#     print(tagged_documents)
#     tags = list(tagged_documents.keys())
#     tag = tags[0]
#     ids_to_documents = atlas.get_documents_by_ids(project=project, ids=tagged_documents[tag])
#     print(f"Num documents tagged with tag `{tag}`: ", len(ids_to_documents))
#     print(list(ids_to_documents.values())[0])
# else:
#     print("No currently tagged documents")
#
# # add a new tag to all of the documents
# print("Adding the tag `my_custom_tag` for all documents.")
# all_documents = atlas.get_documents(project=project)
# atlas.tag_documents(project=project, ids=[doc['_id'] for doc in all_documents], tags=['my_custom_tag'])
#
# newly_tagged_documents = atlas.get_documents_by_tags(project=project)
# print("All document tags:", newly_tagged_documents)
#
#
#
# #remove the tag
# atlas.untag_documents(project=project, ids=[doc['_id'] for doc in all_documents], tags=['my_custom_tag'])
# print("Removed the tag `my_custom_tag` for all documents.")
#
#
#
#
