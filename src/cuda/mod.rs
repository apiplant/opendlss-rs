//! Optional CUDA-driver discovery. CUDA is loaded dynamically so the portable
//! backend remains usable on AMD, Intel, Apple, and software adapters.

pub mod driver;
pub mod kernels;
pub mod network;
pub mod weights;

use anyhow::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaAvailability {
    Available { devices: i32, driver_version: i32 },
    Unavailable(String),
}

/// Probes the CUDA Driver API without creating a CUDA context or loading model
/// code. A CUDA execution backend can use this result to opt in at runtime.
pub fn probe() -> Result<CudaAvailability> {
    #[cfg(target_os = "windows")]
    const NAMES: &[&str] = &["nvcuda.dll"];
    #[cfg(not(target_os = "windows"))]
    const NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];

    let library = NAMES
        .iter()
        .find_map(|name| unsafe { libloading::Library::new(name).ok() });
    let Some(library) = library else {
        return Ok(CudaAvailability::Unavailable(
            "CUDA driver library was not found".into(),
        ));
    };
    unsafe {
        type CuInit = unsafe extern "C" fn(u32) -> i32;
        type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> i32;
        type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
        let init = match library.get::<CuInit>(b"cuInit\0") {
            Ok(symbol) => symbol,
            Err(_) => {
                return Ok(CudaAvailability::Unavailable(
                    "CUDA driver has no cuInit symbol".into(),
                ));
            }
        };
        if init(0) != 0 {
            return Ok(CudaAvailability::Unavailable("cuInit failed".into()));
        }
        let count = match library.get::<CuDeviceGetCount>(b"cuDeviceGetCount\0") {
            Ok(symbol) => symbol,
            Err(_) => {
                return Ok(CudaAvailability::Unavailable(
                    "CUDA driver has no cuDeviceGetCount symbol".into(),
                ));
            }
        };
        let version = match library.get::<CuDriverGetVersion>(b"cuDriverGetVersion\0") {
            Ok(symbol) => symbol,
            Err(_) => {
                return Ok(CudaAvailability::Unavailable(
                    "CUDA driver has no cuDriverGetVersion symbol".into(),
                ));
            }
        };
        let mut devices = 0;
        let mut driver_version = 0;
        if count(&mut devices) != 0 || version(&mut driver_version) != 0 {
            return Ok(CudaAvailability::Unavailable(
                "CUDA device query failed".into(),
            ));
        }
        Ok(CudaAvailability::Available {
            devices,
            driver_version,
        })
    }
}
