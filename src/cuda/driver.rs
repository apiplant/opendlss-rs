//! The slice of the CUDA driver API the backend uses, loaded at run time (as `probe` does), so the crate
//! builds and runs without CUDA and only this backend needs it.

use anyhow::{Context as _, Result, bail};
use std::{
    ffi::{CString, c_char, c_void},
    sync::Arc,
};

pub type CuResult = i32;
pub type DevicePtr = u64;
type Handle = *mut c_void;

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
const CU_STREAM_NON_BLOCKING: u32 = 1;
const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;

macro_rules! driver_api {
    ($($name:ident: fn($($arg:ty),*);)*) => {
        #[allow(non_snake_case)]
        struct Api {
            _library: libloading::Library,
            $($name: unsafe extern "C" fn($($arg),*) -> CuResult,)*
        }

        impl Api {
            fn load() -> Result<Self> {
                #[cfg(target_os = "windows")]
                const NAMES: &[&str] = &["nvcuda.dll"];
                #[cfg(not(target_os = "windows"))]
                const NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];
                let library = NAMES
                    .iter()
                    .find_map(|name| unsafe { libloading::Library::new(name).ok() })
                    .context("the CUDA driver library was not found")?;
                unsafe {
                    Ok(Self {
                        $($name: *library
                            .get(concat!(stringify!($name), "\0").as_bytes())
                            .with_context(|| concat!("the CUDA driver has no ", stringify!($name)))?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}

driver_api! {
    cuInit: fn(u32);
    cuGetErrorString: fn(CuResult, *mut *const c_char);
    cuDeviceGet: fn(*mut i32, i32);
    cuDeviceGetName: fn(*mut c_char, i32, i32);
    cuDeviceGetAttribute: fn(*mut i32, i32, i32);
    cuDevicePrimaryCtxRetain: fn(*mut Handle, i32);
    cuCtxSetCurrent: fn(Handle);
    cuMemAlloc_v2: fn(*mut DevicePtr, usize);
    cuMemFree_v2: fn(DevicePtr);
    cuMemAllocHost_v2: fn(*mut *mut c_void, usize);
    cuMemFreeHost: fn(*mut c_void);
    cuMemcpyHtoD_v2: fn(DevicePtr, *const c_void, usize);
    cuMemcpyDtoH_v2: fn(*mut c_void, DevicePtr, usize);
    cuMemcpyHtoDAsync_v2: fn(DevicePtr, *const c_void, usize, Handle);
    cuMemcpyDtoHAsync_v2: fn(*mut c_void, DevicePtr, usize, Handle);
    cuMemsetD8_v2: fn(DevicePtr, u8, usize);
    cuMemsetD8Async: fn(DevicePtr, u8, usize, Handle);
    cuMemcpyDtoDAsync_v2: fn(DevicePtr, DevicePtr, usize, Handle);
    cuModuleLoadData: fn(*mut Handle, *const c_void);
    cuModuleGetFunction: fn(*mut Handle, Handle, *const c_char);
    cuFuncSetAttribute: fn(Handle, i32, i32);
    cuLaunchKernel: fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut *mut c_void);
    cuStreamCreate: fn(*mut Handle, u32);
    cuStreamSynchronize: fn(Handle);
    cuStreamBeginCapture_v2: fn(Handle, i32);
    cuStreamEndCapture: fn(Handle, *mut Handle);
    cuGraphInstantiateWithFlags: fn(*mut Handle, Handle, u64);
    cuGraphLaunch: fn(Handle, Handle);
    cuGraphExecDestroy: fn(Handle);
    cuGraphDestroy: fn(Handle);
    cuEventCreate: fn(*mut Handle, u32);
    cuEventRecord: fn(Handle, Handle);
    cuEventSynchronize: fn(Handle);
    cuEventElapsedTime: fn(*mut f32, Handle, Handle);
}

pub struct Driver {
    api: Api,
}

impl Driver {
    fn check(&self, result: CuResult, what: &str) -> Result<()> {
        if result == 0 {
            return Ok(());
        }
        let mut message: *const c_char = std::ptr::null();
        unsafe { (self.api.cuGetErrorString)(result, &mut message) };
        let text = if message.is_null() {
            format!("error {result}")
        } else {
            unsafe { std::ffi::CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        bail!("{what}: {text}")
    }
}

/// One device, its primary context, and one stream everything runs on in order.
pub struct Cuda {
    driver: Arc<Driver>,
    context: Handle,
    pub stream: Handle,
    pub name: String,
    pub sm_count: u32,
}

// The context and stream are only driver handles; the driver API is thread-safe.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

impl Cuda {
    /// The first CUDA device, which must be Ada (sm_89) or newer: the PTX kernels use its FP8 MMA.
    pub fn new() -> Result<Self> {
        let driver = Arc::new(Driver { api: Api::load()? });
        let api = &driver.api;
        unsafe {
            driver.check((api.cuInit)(0), "cuInit")?;
            let mut device = 0;
            driver.check((api.cuDeviceGet)(&mut device, 0), "no CUDA device")?;
            let mut name = [0 as c_char; 256];
            driver.check(
                (api.cuDeviceGetName)(name.as_mut_ptr(), 256, device),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let attribute = |which| -> Result<i32> {
                let mut value = 0;
                driver.check(
                    (api.cuDeviceGetAttribute)(&mut value, which, device),
                    "cuDeviceGetAttribute",
                )?;
                Ok(value)
            };
            let (major, minor) = (
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            );
            if (major, minor) < (8, 9) {
                bail!(
                    "{name} is sm_{major}{minor}; the CUDA kernels need FP8 tensor cores (sm_89, Ada, or newer)"
                );
            }
            let sm_count = attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
            let mut context = std::ptr::null_mut();
            driver.check(
                (api.cuDevicePrimaryCtxRetain)(&mut context, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            driver.check((api.cuCtxSetCurrent)(context), "cuCtxSetCurrent")?;
            let mut stream = std::ptr::null_mut();
            driver.check(
                (api.cuStreamCreate)(&mut stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
            )?;
            Ok(Self {
                driver,
                context,
                stream,
                name,
                sm_count,
            })
        }
    }

    fn api(&self) -> &Api {
        &self.driver.api
    }

    fn check(&self, result: CuResult, what: &str) -> Result<()> {
        self.driver.check(result, what)
    }

    /// Makes this context current on the calling thread.
    pub fn bind(&self) -> Result<()> {
        self.check(
            unsafe { (self.api().cuCtxSetCurrent)(self.context) },
            "cuCtxSetCurrent",
        )
    }

    /// A zeroed device allocation.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer> {
        let size = bytes.max(16).next_multiple_of(16);
        let mut ptr = 0;
        self.check(
            unsafe { (self.api().cuMemAlloc_v2)(&mut ptr, size) },
            &format!("allocating {size} bytes"),
        )?;
        self.check(
            unsafe { (self.api().cuMemsetD8_v2)(ptr, 0, size) },
            "zeroing an allocation",
        )?;
        Ok(DeviceBuffer {
            driver: self.driver.clone(),
            ptr,
            size,
        })
    }

    pub fn upload(&self, data: &[u8]) -> Result<DeviceBuffer> {
        let buffer = self.alloc(data.len())?;
        self.write(&buffer, 0, data)?;
        Ok(buffer)
    }

    pub fn write(&self, buffer: &DeviceBuffer, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > buffer.size {
            bail!("write past the end of a device buffer");
        }
        self.check(
            unsafe {
                (self.api().cuMemcpyHtoD_v2)(
                    buffer.ptr + offset as u64,
                    data.as_ptr().cast(),
                    data.len(),
                )
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn read(&self, ptr: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        self.synchronize()?;
        let mut data = vec![0u8; bytes];
        self.check(
            unsafe { (self.api().cuMemcpyDtoH_v2)(data.as_mut_ptr().cast(), ptr, bytes) },
            "cuMemcpyDtoH",
        )?;
        Ok(data)
    }

    /// Page-locked host memory, so the per-frame copies run at full bus speed and asynchronously.
    pub fn host_buffer(&self, bytes: usize) -> Result<HostBuffer> {
        let mut ptr = std::ptr::null_mut();
        self.check(
            unsafe { (self.api().cuMemAllocHost_v2)(&mut ptr, bytes.max(16)) },
            "cuMemAllocHost",
        )?;
        Ok(HostBuffer {
            driver: self.driver.clone(),
            ptr: ptr.cast(),
            len: bytes,
        })
    }

    pub fn copy_to_device_async(
        &self,
        dst: DevicePtr,
        src: &HostBuffer,
        bytes: usize,
    ) -> Result<()> {
        self.check(
            unsafe { (self.api().cuMemcpyHtoDAsync_v2)(dst, src.ptr.cast(), bytes, self.stream) },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn copy_to_host_async(
        &self,
        dst: &mut HostBuffer,
        src: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        self.check(
            unsafe { (self.api().cuMemcpyDtoHAsync_v2)(dst.ptr.cast(), src, bytes, self.stream) },
            "cuMemcpyDtoHAsync",
        )
    }

    pub fn memset_async(&self, ptr: DevicePtr, bytes: usize) -> Result<()> {
        self.check(
            unsafe { (self.api().cuMemsetD8Async)(ptr, 0, bytes, self.stream) },
            "cuMemsetD8Async",
        )
    }

    pub fn copy_async(&self, dst: DevicePtr, src: DevicePtr, bytes: usize) -> Result<()> {
        self.check(
            unsafe { (self.api().cuMemcpyDtoDAsync_v2)(dst, src, bytes, self.stream) },
            "cuMemcpyDtoDAsync",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        self.check(
            unsafe { (self.api().cuStreamSynchronize)(self.stream) },
            "the frame failed",
        )
    }

    /// JIT-compiles a PTX module for this device.
    pub fn module(&self, ptx: &str) -> Result<Module> {
        let text = CString::new(ptx).context("PTX text contains a NUL")?;
        let mut module = std::ptr::null_mut();
        self.check(
            unsafe { (self.api().cuModuleLoadData)(&mut module, text.as_ptr().cast()) },
            "loading PTX",
        )?;
        Ok(Module(module))
    }

    pub fn function(&self, module: &Module, entry: &str) -> Result<Function> {
        let name = CString::new(entry)?;
        let mut function = std::ptr::null_mut();
        self.check(
            unsafe { (self.api().cuModuleGetFunction)(&mut function, module.0, name.as_ptr()) },
            &format!("no kernel {entry}"),
        )?;
        Ok(Function {
            handle: function,
            max_dynamic_shared: 0,
        })
    }

    /// Lets `function` use `shared` bytes of dynamic shared memory; past 48 KB a kernel has to opt in.
    pub fn allow_shared(&self, function: &mut Function, shared: u32) -> Result<()> {
        if shared > 48 * 1024 && shared > function.max_dynamic_shared {
            self.check(
                unsafe {
                    (self.api().cuFuncSetAttribute)(
                        function.handle,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared as i32,
                    )
                },
                "raising the dynamic shared memory limit",
            )?;
            function.max_dynamic_shared = shared;
        }
        Ok(())
    }

    /// Launches `function` on the stream (after `allow_shared` for more than 48 KB).
    pub fn launch(
        &self,
        function: &Function,
        grid: [u32; 3],
        block: u32,
        shared: u32,
        args: &[Arg],
    ) -> Result<()> {
        let mut values: Vec<u64> = args.iter().map(|a| a.bits()).collect();
        let mut pointers: Vec<*mut c_void> =
            values.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        self.check(
            unsafe {
                (self.api().cuLaunchKernel)(
                    function.handle,
                    grid[0],
                    grid[1],
                    grid[2],
                    block,
                    1,
                    1,
                    shared,
                    self.stream,
                    pointers.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Records everything `record` issues on the stream into a CUDA graph, replayable with one launch.
    pub fn capture(&self, record: impl FnOnce() -> Result<()>) -> Result<Graph> {
        self.check(
            unsafe {
                (self.api().cuStreamBeginCapture_v2)(
                    self.stream,
                    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "cuStreamBeginCapture",
        )?;
        let recorded = record();
        let mut graph = std::ptr::null_mut();
        let ended = self.check(
            unsafe { (self.api().cuStreamEndCapture)(self.stream, &mut graph) },
            "cuStreamEndCapture",
        );
        recorded?;
        ended?;
        let mut exec = std::ptr::null_mut();
        let instantiated = self.check(
            unsafe { (self.api().cuGraphInstantiateWithFlags)(&mut exec, graph, 0) },
            "cuGraphInstantiate",
        );
        unsafe { (self.api().cuGraphDestroy)(graph) };
        instantiated?;
        Ok(Graph {
            driver: self.driver.clone(),
            exec,
        })
    }

    pub fn launch_graph(&self, graph: &Graph) -> Result<()> {
        self.check(
            unsafe { (self.api().cuGraphLaunch)(graph.exec, self.stream) },
            "cuGraphLaunch",
        )
    }

    /// Milliseconds the stream spends in `work`.
    pub fn time(&self, work: impl FnOnce() -> Result<()>) -> Result<f64> {
        let api = self.api();
        let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
        unsafe {
            self.check((api.cuEventCreate)(&mut start, 0), "cuEventCreate")?;
            self.check((api.cuEventCreate)(&mut end, 0), "cuEventCreate")?;
            self.check((api.cuEventRecord)(start, self.stream), "cuEventRecord")?;
        }
        work()?;
        let mut milliseconds = 0f32;
        unsafe {
            self.check((api.cuEventRecord)(end, self.stream), "cuEventRecord")?;
            self.check((api.cuEventSynchronize)(end), "the frame failed")?;
            self.check(
                (api.cuEventElapsedTime)(&mut milliseconds, start, end),
                "cuEventElapsedTime",
            )?;
        }
        Ok(milliseconds as f64)
    }
}

/// A kernel argument, passed by value.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    U32(u32),
    F32(f32),
    Ptr(DevicePtr),
}

impl Arg {
    /// The argument's bytes in a u64 slot: cuLaunchKernel reads each through its pointer at its own width.
    fn bits(self) -> u64 {
        match self {
            Arg::U32(v) => v as u64,
            Arg::F32(v) => v.to_bits() as u64,
            Arg::Ptr(v) => v,
        }
    }
}

pub struct Module(Handle);
unsafe impl Send for Module {}

pub struct Function {
    handle: Handle,
    max_dynamic_shared: u32,
}
unsafe impl Send for Function {}

pub struct DeviceBuffer {
    driver: Arc<Driver>,
    pub ptr: DevicePtr,
    pub size: usize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { (self.driver.api.cuMemFree_v2)(self.ptr) };
    }
}

pub struct HostBuffer {
    driver: Arc<Driver>,
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe { (self.driver.api.cuMemFreeHost)(self.ptr.cast()) };
    }
}

pub struct Graph {
    driver: Arc<Driver>,
    exec: Handle,
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { (self.driver.api.cuGraphExecDestroy)(self.exec) };
    }
}
