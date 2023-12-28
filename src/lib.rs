use cudarc::driver::sys;
use cudarc::{
    driver::{CudaDevice, CudaSlice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use glob::glob;
use lazy_static::lazy_static;
use serde_derive::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::{env, path::PathBuf};

lazy_static! {
    static ref TRITON_CACHE: String = {
        match env::var("TRITON_CACHE_DIR") {
            Ok(val) => val,
            Err(_e) => {
                let home = dirs::home_dir().unwrap();
                format!("{}/.triton/cache", home.display())
            }
        }
    };
}

static KERNEL_NAME: &str = "add_kernel_0d1d2d3de";
pub struct TritonKernel {
    pub name: String,
}

impl TritonKernel {
    pub fn new(name: &str) -> TritonKernel {
        TritonKernel {
            name: name.to_string(),
        }
    }
    pub fn metadata(&self) -> TritonMetadata {
        let paths = find_ext(&self.name, "json");
        assert!(paths.len() == 1);
        let mut file = File::open(&paths[0]).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let metadata: TritonMetadata = match serde_json::from_str(&contents) {
            Ok(v) => v,
            Err(e) => panic!("{:?}", e),
        };
        metadata
    }
    pub fn cubin(&self) -> Vec<PathBuf> {
        find_ext(&self.name, "cubin")
    }
    pub fn ptx(&self) -> Vec<PathBuf> {
        find_ext(&self.name, "ptx")
    }
}
pub fn find_ext(kernel_name: &str, ext: &str) -> Vec<PathBuf> {
    let mut found_paths: Vec<PathBuf> = Vec::new();
    for p in glob(&format!("{}/**/{}.{}", *TRITON_CACHE, kernel_name, ext))
        .expect("Unable to parse glob")
    {
        match p {
            Ok(p) => {
                found_paths.push(p);
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
    found_paths
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TritonMetadata {
    target: Vec<Value>,
    num_warps: u32,
    num_ctas: u32,
    num_stages: u32,
    cluster_dims: Vec<u32>,
    ptx_version: Option<u32>,
    enable_warp_specialization: bool,
    enable_persistent: bool,
    optimize_epilogue: bool,
    enable_fp_fusion: bool,
    allow_fp8e4nv: bool,
    max_num_imprecise_acc_default: u32,
    extern_libs: Option<Vec<String>>,
    debug: Option<bool>,
    AMDGCN_ENABLE_DUMP: bool,
    DISABLE_FAST_REDUCTION: bool,
    DISABLE_MMA_V3: bool,
    ENABLE_TMA: bool,
    LLVM_IR_ENABLE_DUMP: bool,
    MLIR_ENABLE_DUMP: bool,
    TRITON_DISABLE_LINE_INFO: bool,
    ids_of_folded_args: Vec<u32>,
    ids_of_tensormaps: Option<Vec<u32>>,
    shared: u32,
    name: String,
}
impl TritonMetadata {
    pub fn target(&self) -> String {
        self.target
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join("")
            .replace('\"', "")
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let kernel = TritonKernel::new("add_kernel");
        let paths = kernel.cubin();
        assert!(paths.len() == 1);
        let paths = kernel.ptx();
        assert!(paths.len() == 1);
    }

    #[test]
    fn test_metadata() {
        let kernel = TritonKernel::new("add_kernel");
        let metadata = kernel.metadata();
        assert!(metadata.name.contains(format!("{}", kernel.name).as_str()));
    }

    #[test]
    fn test_kernel() -> Result<(), DriverError> {
        let kernel = TritonKernel::new("add_kernel");
        // You can load a function from a pre-compiled PTX like so:
        let ptx_path = &kernel.ptx()[0];
        let cubin_path = &kernel.cubin()[0];
        let metadata = kernel.metadata();
        let num_threads = metadata.num_warps * 32;
        // let func_name: String = metadata.name.clone();
        let module_name = "triton";

        let dev = CudaDevice::new(0)?;
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
        let cfg = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (num_threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, n)) }?;

        let c_host = dev.sync_reclaim(c_dev)?;
        assert!(c_host == expected);

        Ok(())
    }
}
