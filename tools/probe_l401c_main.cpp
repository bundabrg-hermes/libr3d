// probe_l401c_main.cpp -- dlopen the probe .so and call its run_probe() (kernel launches
// happen inside a dlopen'd module, exactly like the plugin).
#include <dlfcn.h>
#include <cstdio>
int main() {
    void* dl = dlopen("/tmp/libprobe41.so", RTLD_NOW);
    if (!dl) { printf("dlopen: %s\n", dlerror()); return 1; }
    auto f = (int (*)())dlsym(dl, "run_probe");
    if (!f) { printf("dlsym: %s\n", dlerror()); return 1; }
    return f();
}
