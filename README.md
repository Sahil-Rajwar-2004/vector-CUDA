# Vector calculation in C++ with GPU acceleration using CUDA

## Requirements
- GPU (obvious)
- nvcc compiler
- C++/C compiler

```cu
#include <iostream>
#include <vector_cuda.h>
using namespace std;

int main(){
    Vector vec1(1,2,3);
    Vector vec2(2,3,4);

    Vector add = vec1.add(vec2);
    Vector sub = vec1.sub(vec2);

    add.print();
    sub.print();
}
```

```bash
vector: (3, 5, 7)
vector:(-1, -1, -1)
```


