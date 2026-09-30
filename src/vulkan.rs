use anyhow::{Context, Result};
use ash::{Entry, vk};
use std::collections::BTreeSet;

pub const REQUIRED_EXTENSIONS: &[&str] = &[
    "VK_KHR_cooperative_matrix",
    "VK_NV_cooperative_matrix2",
    "VK_EXT_shader_float8",
    "VK_NV_cuda_kernel_launch",
];

#[derive(Debug)]
pub struct DeviceReport {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub compute_queues: u32,
    pub missing_extensions: Vec<&'static str>,
}
impl DeviceReport {
    pub fn supported(&self) -> bool {
        self.missing_extensions.is_empty()
    }
}

pub fn inspect_devices() -> Result<Vec<DeviceReport>> {
    let entry = unsafe { Entry::load() }.context("cannot load the Vulkan loader")?;
    let app = std::ffi::CString::new("opendlss-nr")?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app)
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    let instance =
        unsafe { entry.create_instance(&info, None) }.context("cannot create Vulkan instance")?;
    let result = (|| -> Result<Vec<DeviceReport>> {
        let physical = unsafe { instance.enumerate_physical_devices() }
            .context("cannot enumerate Vulkan devices")?;
        physical
            .into_iter()
            .map(|device| {
                let properties = unsafe { instance.get_physical_device_properties(device) };
                let name = unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                let extensions = unsafe { instance.enumerate_device_extension_properties(device) }
                    .context("cannot enumerate device extensions")?
                    .into_iter()
                    .map(|e| {
                        unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect::<BTreeSet<_>>();
                let missing_extensions = REQUIRED_EXTENSIONS
                    .iter()
                    .copied()
                    .filter(|needed| !extensions.contains(*needed))
                    .collect();
                let compute_queues =
                    unsafe { instance.get_physical_device_queue_family_properties(device) }
                        .iter()
                        .filter(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                        .map(|q| q.queue_count)
                        .sum();
                Ok(DeviceReport {
                    name,
                    vendor_id: properties.vendor_id,
                    device_id: properties.device_id,
                    compute_queues,
                    missing_extensions,
                })
            })
            .collect()
    })();
    unsafe { instance.destroy_instance(None) };
    result
}
