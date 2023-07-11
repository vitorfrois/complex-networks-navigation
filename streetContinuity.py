import networkx as nx

def toDualGraphWithContinuity(G):
  addedNodes = {
  }

  correspondence = {}

  nG = nx.graph.Graph()

  for edge in G.edges(data=True):
      n1, n2, data = edge

      if n1 not in correspondence:
          correspondence[n1] = {}
      if n2 not in correspondence:
          correspondence[n2] = {}

      osmids = [data['osmid']] if type(data['osmid']) != list else data['osmid']
      names = [data.get('name', None)] if type(data.get('name', None)) != list else data.get('name', None)

      for osmid, name in zip(osmids, names):
          if osmid not in addedNodes:
              x = G.nodes[n1]['x'] + (G.nodes[n2]['x'] - G.nodes[n1]['x']) / 2
              y = G.nodes[n1]['y'] + (G.nodes[n2]['y'] - G.nodes[n1]['y']) / 2
              lat = G.nodes[n1]['lat'] + (G.nodes[n2]['lat'] - G.nodes[n1]['lat']) / 2
              lon = G.nodes[n1]['lon'] + (G.nodes[n2]['lon'] - G.nodes[n1]['lon']) / 2

              nG.add_node(osmid, name=name, x=x, y=y, lat=lat, lon=lon)
              addedNodes[osmid] = True

          correspondence[n1][osmid] = True
          correspondence[n2][osmid]  = True
          
  for node in G.nodes():
      osmids = list(correspondence.get(node, {}).keys())

      for i in range(len(osmids)):
          for j in range(i+1, len(osmids)):
              nG.add_edge(osmids[i], osmids[j])
  
  return nG

