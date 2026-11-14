# project(mlx.data) runs before mlx-c declares mlxc; patch the target once it exists.
function(mlx_sys_cuda_half)
  target_compile_options(mlxc PRIVATE "SHELL:-include ${MLX_SYS_CUDA_HALF}")
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL mlx_sys_cuda_half)

# MLX's CUDA caches (conv plans, SDPA, graphs) throw once they have missed more than twice their
# capacity over the process's life: a server seeing many shapes then fails every request. Make
# them evict only (the source is fetched by now and compiled after configure).
function(mlx_sys_cuda_no_thrash)
  set(header "${mlx_SOURCE_DIR}/mlx/backend/cuda/lru_cache.h")
  file(READ "${header}" text)
  set(check "if (env_name_ && ++cache_misses_ > 2 * capacity_) {")
  string(FIND "${text}" "${check}" at)
  if(at EQUAL -1)
    string(FIND "${text}" "mlx_sys: never throw" done)
    if(done EQUAL -1)
      message(FATAL_ERROR "mlx-sys: ${header} changed, review the cache-thrashing patch")
    endif()
    return()
  endif()
  string(REPLACE "${check}" "if (false) { // mlx_sys: never throw on misses, evict instead" text "${text}")
  file(WRITE "${header}" "${text}")
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL mlx_sys_cuda_no_thrash)
