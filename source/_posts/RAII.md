---
title: C++ 中的 RAII 介绍
top: false
cover: false
toc: true
mathjax: false
date: 2019-10-13 11:15:21
password:
summary:
tags: C++
categories: C++
---

<!-- TOC -->

- [什么是 RAII？](#什么是-raii)
- [资源管理](#资源管理)
- [状态管理](#状态管理)
- [总结](#总结)

<!-- /TOC -->

# 什么是 RAII？

- RAII 是 C++ 的发明者 Bjarne Stroustrup 提出的概念，全称是 “Resource Acquisition is Initialization”，直译过来是“资源获取即初始化”
- 在构造函数中申请分配资源，在析构函数中释放资源。因为C++的语言机制保证了，当一个对象创建的时候，自动调用构造函数，当对象超出作用域的时候会自动调用析构函数。
- 在 RAII 的指导下，我们应该使用类来管理资源，将资源和对象的生命周期绑定。

# 资源管理

- 智能指针（`std::shared_ptr` 和 `std::unique_ptr`）即 RAII 最具代表的实现，使用智能指针，可以实现自动的内存管理，再也不需要担心忘记 `delete` 造成的内存泄漏。
- 一个简单的 RAII 实现

```c++
#include <iostream>
#include <functional>
#include <fstream>

#define SCOPEGUARD_LINENAME(name, line) name##line
#define ON_SCOPE_EXIT(callback) ScopeGuard SCOPEGUARD_LINENAME(EXIT, __LINE__)(callback)

class ScopeGuard {
public:
    explicit ScopeGuard(std::function<void()> f) :
        handle_exit_scope_(f){}

    ~ScopeGuard(){ handle_exit_scope_(); }
private:
    std::function<void()> handle_exit_scope_;
};

class A {
public:
    A() {
        std::cout << "A()" << std::endl;
    }

    ~A() {
        std::cout << "~A()" << std::endl;
    }
};

int main(int argc, char** argv) {
    {
        A *a = new A();
        ON_SCOPE_EXIT([&] {delete a; });
        // ......
    }

    {
        std::ofstream f("test.txt");
        ON_SCOPE_EXIT([&] {f.close(); });
        // ......
    }

    return 0;
}
```

- 当 `ScopeGuard` 对象超出作用域，`ScopeGuard` 的析构函数中会调用 `handle_exit_scope_` 函数，也就是 `lambda` 表达式中的内容，所以在 `lamabda` 表达式中填上资源释放的代码即可，简洁、明了。

# 状态管理

- RAII 另一个引申的应用是可以实现安全的状态管理。一个典型的应用就是在线程同步中，使用 `std::unique_lock` 或者 `std::lock_guard` 对互斥量 `std::mutex` 进行状态管理。通常我们不会写出如下的代码：

```c++
std::mutex mutex_;
void function() {
    mutex_.lock();
    // ......
    // ......
    mutex_.unlock();
}
```

- 在互斥量 `lock` 和 `unlock` 之间的代码很可能会出现异常，或者有 `return` 语句，这样的话，互斥量就不会正确的 `unlock`，会导致线程的死锁。
- 正确的方式是使用 `std::unique_lock` 或者 `std::lock_guard` 对互斥量进行状态管理

```cpp
std::mutex mutex_;
void function() {
    std::lock_guard<std::mutex> lock(mutex_);
    // ......
    // ......
}
```

- 在创建 `std::lock_guard` 对象的时候，会对 `std::mutex` 对象进行 `lock`，当 `std::lock_guard` 对象在超出作用域时，会自动对 `std::mutex` 对象进行解锁，这样的话，就不用担心代码异常造成的线程死锁。

# 总结

通过上面的分析可以看出，RAII 的核心思想是将资源或者状态与对象的生命周期绑定，通过 C++ 的语言机制，实现资源和状态的安全管理。理解和使用RAII能使软件设计更清晰，代码更健壮。
