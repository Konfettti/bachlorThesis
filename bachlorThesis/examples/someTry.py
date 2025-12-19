import featuretools as ft

primitives = ft.list_primitives()
print(primitives['name'].tolist())
