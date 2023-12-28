### Rust Triton Kernel Minimal Example
Minimal example of loading a `triton` kernel in `Rust`.  

#### Example
The unittests in `lib.rs` demonstrate the `API`.    

A reproduction of the [`add_kernel` tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py):
```rust
//Kernel name should match that of the @triton.jit decorated function
let kernel = TritonKernel::new("add_kernel");

//TritonKernel will search TRITON_CACHE_DIR (~/.triton/cache on Linux systems) for the following  
let cubin_path = &kernel.cubin()[0];
let metadata = kernel.metadata();
let num_threads = metadata.num_warps * 32;

/*Load kernel
- Note that loading PTX instead of CUBIN raises incompatible PTX version error for some reason.
- `cudarc` supports loading ptx / cubin interchangeably since it calls to cuModuleLoad under the hood, which accepts either.
*/
let module_name = "triton";
let dev = CudaDevice::new(0)?;

/*KERNEL_NAME here is the name of the embedded cuda kernel, 
which is the original kernel name ("add_kernel") with a suffix determined by compiler specializations.  
- This mangled name is available in the metadata, but `cudarc` `load_ptx` requires this to be a static string.  
- Need a more ergonomic way of setting the kernel func name -- currently a hard-coded static string.   
*/
dev.load_ptx(Ptx::from_file(cubin_path), module_name, &[KERNEL_NAME])?;
let f = dev.get_func(module_name, KERNEL_NAME).unwrap();

let a_host: [f32; 3] = [1.0, 2.0, 3.0];
let b_host: [f32; 3] = [4.0, 5.0, 6.0];

let a_dev = dev.htod_copy(a_host.into())?;
let b_dev = dev.htod_copy(b_host.into())?;
let expected = a_host
    .into_iter()
    .zip(b_host)
    .map(|(a, b)| a + b)
    .collect::<Vec<_>>();
let mut c_dev: CudaSlice<f32> = dev.alloc_zeros(a_host.len()).unwrap();

let n = a_host.len() as u32;
 
let block_dim_x = n.div_ceil(num_threads);
let cfg = LaunchConfig {
    block_dim: (block_dim_x, 1, 1),
    grid_dim: (num_threads, 1, 1),
    shared_mem_bytes: 0,
};
unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, n)) }?;

let c_host = dev.sync_reclaim(c_dev)?;
assert!(c_host == expected);
```

#### Notes
Triton appends a suffix to the original kernel name (the function decorated by `@triton.jit`) in the embedded cubin.  This name is stored in the metadata generated along with the cubin.  However, since the `cuFunction` name in the `cudarc` `load_ptx` needs to be a `static &str`, this mangled name can not be loaded at runtime.  

I've hardcoded the mangled name but imagine this can be done in `build.rs`. 
