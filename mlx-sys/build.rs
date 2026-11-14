extern crate cmake;

use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

#[path = "src/bindgen_config.rs"]
mod bindgen_config;

#[path = "src/native_link.rs"]
mod native_link;

#[cfg(all(feature = "cuda", any(feature = "metal", feature = "accelerate")))]
compile_error!(
    "mlx-sys: `cuda` cannot be combined with `metal` or `accelerate`; \
     use `default-features = false, features = [\"cuda\"]`"
);

#[cfg(feature = "metal")]
fn mlx_c_key(mlx_c_root: &Path) -> String {
    if mlx_c_root.join(".git").exists() {
        let output = Command::new("git")
            .arg("-C")
            .arg(mlx_c_root)
            .args(["rev-parse", "HEAD"])
            .output();
        if let Ok(output) = output {
            if output.status.success() {
                let commit = String::from_utf8_lossy(&output.stdout);
                let commit = commit.trim();
                if commit.len() >= 12 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return commit[..12].to_owned();
                }
            }
        }
    }

    let mut files = bindgen_config::discover_headers(mlx_c_root)
        .expect("Unable to discover mlx-c headers for the metallib key");
    files.push(mlx_c_root.join("CMakeLists.txt"));
    files.sort();

    let mut hash = 0xcbf29ce484222325_u64;
    for file in files {
        let relative = file.strip_prefix(mlx_c_root).unwrap_or(&file);
        for byte in relative.to_string_lossy().bytes().chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for byte in fs::read(&file).expect("Unable to hash mlx-c source for the metallib key") {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")[..12].to_owned()
}

#[cfg(feature = "metal")]
fn metallib_dir(mlx_c_root: &Path) -> PathBuf {
    if let Some(path) = env::var_os("MLX_RS_METAL_PATH") {
        return PathBuf::from(path);
    }

    let home =
        env::var_os("HOME").expect("HOME must be set when MLX_RS_METAL_PATH is not provided");
    // A JIT metallib holds only a few kernels; sharing the directory would break
    // non-JIT builds of the same mlx-c.
    let key = if cfg!(feature = "metal-jit") {
        format!("{}-jit", mlx_c_key(mlx_c_root))
    } else {
        mlx_c_key(mlx_c_root)
    };
    PathBuf::from(home).join(".mlx").join("lib").join(key)
}

/// `CUDA_HOME` or `CUDA_PATH` when set, else `PATH`, else the default toolkit location.
#[cfg(feature = "cuda")]
fn find_nvcc() -> Result<PathBuf, String> {
    if let Some((name, root)) = ["CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .find_map(|name| env::var_os(name).map(|root| (name, PathBuf::from(root))))
    {
        let nvcc = root.join("bin").join("nvcc");
        return nvcc
            .is_file()
            .then_some(nvcc)
            .ok_or_else(|| format!("{name}={} has no bin/nvcc", root.display()));
    }
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .chain([PathBuf::from("/usr/local/cuda/bin")])
        .map(|dir| dir.join("nvcc"))
        .find(|nvcc| nvcc.is_file())
        .ok_or_else(|| "nvcc not found; install the CUDA toolkit or set CUDA_HOME".to_owned())
}

#[cfg(feature = "cuda")]
fn configure_cuda(config: &mut Config) -> Result<(), String> {
    let target = env::var("TARGET").unwrap_or_default();
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return Err(format!(
            "MLX's CUDA backend builds only for Linux targets, not {target}"
        ));
    }
    config.define("MLX_BUILD_CUDA", "ON");
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");
    config.define("CMAKE_CUDA_COMPILER", find_nvcc()?);
    // Unset: MLX queries the GPU of the build machine.
    if let Some(architectures) = env::var_os("MLX_CUDA_ARCHITECTURES").filter(|a| !a.is_empty()) {
        config.define("MLX_CUDA_ARCHITECTURES", architectures);
    }
    // Shared libmlx carries its CUDA, cuDNN and BLAS dependencies, and finds its JIT headers
    // in the include directory beside its lib directory (both under build/).
    config.define("BUILD_SHARED_LIBS", "ON");
    config.define("CMAKE_INSTALL_LIBDIR", "lib");
    config.define("CMAKE_INSTALL_RPATH", "$ORIGIN");
    config.define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON");
    // mlx-c declares float16/bfloat16 functions only on aarch64.
    let canonical = |path: &str| {
        Path::new(path)
            .canonicalize()
            .map_err(|e| format!("cannot locate {path}: {e}"))
    };
    config.define("CMAKE_PROJECT_mlx.data_INCLUDE", canonical("cuda.cmake")?);
    config.define("MLX_SYS_CUDA_HALF", canonical("src/cuda_half.h")?);
    Ok(())
}

/// CUDA builds link the installed shared libraries (`build/lib`).
#[cfg(feature = "cuda")]
fn link_shared(dst: &Path) {
    let lib = dst.join("build").join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");
    // Cargo applies link args to this crate's tests and examples only; dependents use DEP_MLX_LIB_DIR.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo::metadata=lib_dir={}", lib.display());
}

fn build_and_link_mlx_c() {
    let mlx_c_root = Path::new("src/mlx-c");
    // A finished build of this crate (its OUT_DIR, with build/lib, build/include and
    // build/_deps/mlx-src): linked as is, without CMake. CUDA builds only.
    #[cfg(feature = "cuda")]
    if let Some(dst) = env::var_os("MLX_SYS_PREBUILT").filter(|d| !d.is_empty()) {
        let dst = PathBuf::from(dst);
        publish_native_sources(&dst, mlx_c_root);
        link_shared(&dst);
        return;
    }
    let mut config = Config::new(mlx_c_root);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
        #[cfg(feature = "metal-jit")]
        config.define("MLX_METAL_JIT", "ON");
        let metallib_dir = metallib_dir(mlx_c_root);
        fs::create_dir_all(&metallib_dir).expect("Unable to create the MLX metallib directory");
        config.define("MLX_METAL_PATH", &metallib_dir);
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    #[cfg(feature = "cuda")]
    if let Err(error) = configure_cuda(&mut config) {
        eprintln!("mlx-sys: cannot configure the CUDA build: {error}");
        std::process::exit(1);
    }

    // build the mlx-c project
    let dst = config.build();

    publish_native_sources(&dst, mlx_c_root);
    #[cfg(feature = "cuda")]
    link_shared(&dst);
    #[cfg(not(feature = "cuda"))]
    link_static(&dst, mlx_c_root);
}

fn link_static(dst: &Path, mlx_c_root: &Path) {
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    // mlx's GGUF io depends on the vendored gguflib archive, which cmake leaves
    // in the mlx build tree instead of installing next to libmlx.
    println!(
        "cargo:rustc-link-search=native={}/build/_deps/mlx-build/mlx/io",
        dst.display()
    );
    for library in native_link::libraries(&dst.join("build")) {
        if library == "jaccl" {
            println!(
                "cargo:rustc-link-search=native={}/build/jaccl",
                dst.display()
            );
        }
        println!("cargo:rustc-link-lib=static={library}");
    }

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        let metallib = metallib_dir(mlx_c_root).join("mlx.metallib");
        if !metallib.exists() {
            println!(
                "cargo:warning=mlx.metallib was not created at {}; Metal operations may fail at runtime",
                metallib.display()
            );
        }
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = native_link::find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

/// Dependents read this metadata as `DEP_MLX_*`; the README documents the contract.
fn publish_native_sources(dst: &Path, mlx_c_root: &Path) {
    let deps = dst.join("build").join("_deps");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include = [
        Some(deps.join("mlx-src")),
        cfg!(feature = "metal").then(|| deps.join("metal_cpp-src")),
        Some(manifest.join(mlx_c_root)),
    ];
    let include = env::join_paths(include.into_iter().flatten())
        .expect("MLX include directories must not contain the path-list separator");
    println!("cargo::metadata=include={}", include.display());
    let defines = if cfg!(feature = "cuda") { "" } else { "MLX_STATIC" };
    println!("cargo::metadata=defines={defines}");
    let version = fs::read_to_string(mlx_c_root.join("CMakeLists.txt"))
        .ok()
        .and_then(|cmake| {
            cmake.lines().find_map(|line| {
                let tag = line.trim().strip_prefix("GIT_TAG v")?;
                tag.split(|c: char| c == ')' || c.is_whitespace())
                    .next()
                    .map(str::to_owned)
            })
        });
    if let Some(version) = version {
        println!("cargo::metadata=version={version}");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_RS_METAL_PATH");
    println!("cargo:rerun-if-changed=native-libraries.txt");
    println!("cargo:rerun-if-changed=src/native_link.rs");
    println!("cargo:rerun-if-changed=cuda.cmake");
    println!("cargo:rerun-if-changed=src/cuda_half.h");
    for var in ["MLX_CUDA_ARCHITECTURES", "CUDA_HOME", "CUDA_PATH", "MLX_SYS_PREBUILT"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    build_and_link_mlx_c();

    let mlx_c_root = PathBuf::from("src/mlx-c");
    let headers =
        bindgen_config::discover_headers(&mlx_c_root).expect("Unable to discover headers");
    for header in &headers {
        let relative =
            bindgen_config::relative_header(&mlx_c_root, header).expect("Unable to record header");
        println!(
            "cargo:rerun-if-changed={}",
            mlx_c_root.join(relative).display()
        );
    }
    let builder = bindgen_config::builder(&mlx_c_root, &headers);
    #[cfg(feature = "cuda")]
    let builder = builder.clang_args(["-include", "src/cuda_half.h"]);
    let bindings = builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
