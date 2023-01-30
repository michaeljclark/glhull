# glhull

_glhull_ is an experiment to render béziergon convex interior hulls.

- loading truetype glyphs into manifold buffers using FreeType.
- division of béziergons into interior and exterior hulls.
- splitting of béziergon interior hulls into convex polygons.
- outputting stanford polygon files and testing for convexity.
- simple debug UI using GLFW and nanovg to trace convex hulls.

![glhull](/images/glhull.png)

## Build Instructions

```
sudo apt-get install -y cmake ninja-build libfreetype-dev
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```
