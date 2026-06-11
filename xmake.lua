add_rules("mode.debug", "mode.release")

add_repositories("my-repo repo")

add_requires(
    "ygopro-core 0.0.2", "lua 5.3.6", "fmt 10.2.*", "glog 0.6.0",
    "sqlite3 3.43.0+200", "concurrentqueue 1.0.4", "unordered_dense 4.4.*",
    "sqlitecpp 3.2.1")

if not is_plat("windows") then
    add_requires("pybind11 2.13.*")
    add_requires("edopro-core")
end

function add_pybind11()
    if is_plat("windows") then
        local pybind11_include = "$(env PYBIND11_INCLUDE)"
        local python_include = "$(env PYTHON_INCLUDE)"
        local python_libdir = "$(env PYTHON_LIBDIR)"
        local python_version = "$(env PYTHON_VERSION_NODOT)"

        add_includedirs(pybind11_include, python_include)
        add_linkdirs(python_libdir)
        add_links("python" .. python_version)
    else
        add_packages("pybind11")
    end
end

function add_windows_compat()
    if is_plat("windows") then
        add_defines("NOMINMAX", "WIN32_LEAN_AND_MEAN", "FMT_CONSTEVAL=")
        set_languages("c++20")
    else
        set_languages("c++17")
    end
end


target("ygopro0_ygoenv")
    add_rules("python.library", {soabi = not is_plat("windows")})
    add_files("ygoenv/ygoenv/ygopro0/*.cpp")
    add_packages("fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core", "lua")
    add_pybind11()
    add_windows_compat()
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro0"
        os.cp(target:targetfile(), install_target)
        if is_plat("windows") then
            os.cp(target:targetfile(), path.join(install_target, target:name() .. ".pyd"))
        end
        print("Copy target to " .. install_target)
    end)


target("ygopro_ygoenv")
    add_rules("python.library", {soabi = not is_plat("windows")})
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core", "lua")
    add_pybind11()
    add_windows_compat()
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro"
        os.cp(target:targetfile(), install_target)
        if is_plat("windows") then
            os.cp(target:targetfile(), path.join(install_target, target:name() .. ".pyd"))
        end
        print("Copy target to " .. install_target)
    end)

if not is_plat("windows") then
    target("edopro_ygoenv")
        add_rules("python.library", {soabi = not is_plat("windows")})
        add_files("ygoenv/ygoenv/edopro/*.cpp")
        add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "edopro-core")
        add_windows_compat()
        if is_mode("release") then
            set_policy("build.optimization.lto", true)
            add_cxxflags("-march=native")
        end
        add_includedirs("ygoenv")

        after_build(function (target)
            local install_target = "$(projectdir)/ygoenv/ygoenv/edopro"
            os.cp(target:targetfile(), install_target)
            print("Copy target to " .. install_target)
        end)
end


target("alphazero_mcts")
    add_rules("python.library", {soabi = not is_plat("windows")})
    add_files("mcts/mcts/alphazero/*.cpp")
    add_pybind11()
    add_windows_compat()
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("mcts")

    after_build(function (target)
        local install_target = "$(projectdir)/mcts/mcts/alphazero"
        os.cp(target:targetfile(), install_target)
        if is_plat("windows") then
            os.cp(target:targetfile(), path.join(install_target, target:name() .. ".pyd"))
        end
        print("Copy target to " .. install_target)
        os.run("pybind11-stubgen mcts.alphazero.alphazero_mcts -o %s", "$(projectdir)/mcts")
    end)
