import numpy as np

directions = [(-1,0),(1,0),(0,-1),(0,1)]

def dfs(x,y,visited,array):
    visited[x,y] = True
    for dx,dy in directions:
        nx,ny = x + dx, y + dy
        if 0 <= nx < visited.shape[0] and 0 <= ny < visited.shape[1]:
            if not visited[nx,ny] and array[nx,ny] != 0:
                visited = dfs(nx,ny,visited,array)
    return visited

def count_islands(array):
    visited = np.zeros(array.shape, dtype=bool)
    island_count = 0

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if array[i,j] != 0 and not visited[i,j]:
                visited = dfs(i,j,visited,array)
                island_count += 1
    return island_count