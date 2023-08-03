// Copyright (C) 2022 Alibaba Cloud. All rights reserved.
// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::ops::Deref;
use std::sync::MutexGuard;

use dbs_arch::gic::GICDevice;
use dbs_arch::pmu::initialize_pmu;
use dbs_arch::{MMIODeviceInfo, VpmuFeatureLevel};
use dbs_boot::fdt_utils::*;
use dbs_boot::InitrdConfig;
use dbs_utils::epoll_manager::EpollManager;
use dbs_utils::time::TimestampUs;
use linux_loader::cmdline::{Cmdline, Error as CmdlineError};
use vm_memory::GuestAddressSpace;
use vmm_sys_util::eventfd::EventFd;

use super::{Vm, VmError};
use crate::address_space_manager::{GuestAddressSpaceImpl, GuestMemoryImpl};
use crate::error::{Error, StartMicroVmError};
use crate::event_manager::EventManager;
use crate::vcpu::VcpuManager;

impl Vm {
    /// Gets a reference to the irqchip of the VM
    pub fn get_irqchip(&self) -> &dyn GICDevice {
        self.irqchip_handle.as_ref().unwrap().as_ref()
    }

    /// Creates the irq chip in-kernel device model.
    pub fn setup_interrupt_controller(&mut self) -> std::result::Result<(), StartMicroVmError> {
        let vcpu_count = self.vm_config.max_vcpu_count;

        self.irqchip_handle = Some(
            dbs_arch::gic::create_gic(&self.vm_fd, vcpu_count.into())
                .map_err(|e| StartMicroVmError::ConfigureVm(VmError::SetupGIC(e)))?,
        );

        Ok(())
    }

    /// Setup pmu devices for guest vm.
    pub fn setup_pmu_devices(&mut self) -> std::result::Result<(), StartMicroVmError> {
        let vm = self.vm_fd();
        let mut vcpu_manager = self.vcpu_manager().map_err(StartMicroVmError::Vcpu)?;
        let vpmu_feature = vcpu_manager.vpmu_feature();
        if vpmu_feature == VpmuFeatureLevel::Disabled {
            return Ok(());
        }

        for vcpu in vcpu_manager.vcpus_mut() {
            initialize_pmu(vm, vcpu.vcpu_fd())
                .map_err(|e| StartMicroVmError::ConfigureVm(VmError::SetupPmu(e)))?;
        }

        Ok(())
    }

    /// Initialize the virtual machine instance.
    ///
    /// It initialize the virtual machine instance by:
    /// 1) Initialize virtual machine reset event fd.
    /// 2) Create and initialize vCPUs.
    /// 3) Create and initialize interrupt controller.
    /// 4) Create and initialize vPMU device.
    /// 5) Create and initialize devices, such as virtio, block, net, vsock, vfio etc.
    pub fn init_microvm(
        &mut self,
        epoll_mgr: EpollManager,
        vm_as: GuestAddressSpaceImpl,
        request_ts: TimestampUs,
    ) -> Result<(), StartMicroVmError> {
        let reset_eventfd =
            EventFd::new(libc::EFD_NONBLOCK).map_err(|_| StartMicroVmError::EventFd)?;
        self.reset_eventfd = Some(
            reset_eventfd
                .try_clone()
                .map_err(|_| StartMicroVmError::EventFd)?,
        );
        self.vcpu_manager()
            .map_err(StartMicroVmError::Vcpu)?
            .set_reset_event_fd(reset_eventfd)
            .map_err(StartMicroVmError::Vcpu)?;

        // On aarch64, the vCPUs need to be created (i.e call KVM_CREATE_VCPU) and configured before
        // setting up the IRQ chip because the `KVM_CREATE_VCPU` ioctl will return error if the IRQCHIP
        // was already initialized.
        // Search for `kvm_arch_vcpu_create` in arch/arm/kvm/arm.c.
        let kernel_loader_result = self.load_kernel(vm_as.memory().deref())?;
        self.vcpu_manager()
            .map_err(StartMicroVmError::Vcpu)?
            .create_boot_vcpus(request_ts, kernel_loader_result.kernel_load)
            .map_err(StartMicroVmError::Vcpu)?;
        self.setup_interrupt_controller()?;
        self.setup_pmu_devices()?;
        self.init_devices(epoll_mgr)?;

        Ok(())
    }

    /// Generate fdt information about VM.
    fn get_fdt_vm_info<'a>(
        &'a self,
        vm_memory: &'a GuestMemoryImpl,
        cmdline: &'a str,
        initrd_config: Option<&'a InitrdConfig>,
        vcpu_manager: &'a MutexGuard<VcpuManager>,
    ) -> FdtVmInfo {
        let guest_memory = vm_memory.memory();
        let vcpu_mpidr = vcpu_manager
            .vcpus()
            .into_iter()
            .map(|cpu| cpu.get_mpidr())
            .collect();
        let vm_config = self.vm_config();
        let mut vcpu_boot_onlined = vec![1; vm_config.vcpu_count as usize];
        vcpu_boot_onlined.resize(vm_config.max_vcpu_count as usize, 0);
        let vpmu_feature = vcpu_manager.vpmu_feature();
        // This configuration is used for passing cache information into guest.
        let cache_passthrough_enabled = vm_config.enable_cache_passthrough
            && !vm_config.numa_regions.is_empty()
            && vm_config
                .numa_regions
                .iter()
                .all(|info| info.pcpu_ids.is_some());
        let fdt_vcpu_info = FdtVcpuInfo::new(
            vcpu_mpidr,
            vcpu_boot_onlined,
            vpmu_feature,
            cache_passthrough_enabled,
        );

        FdtVmInfo::new(guest_memory, cmdline, initrd_config, fdt_vcpu_info)
    }

    // This method is used for passing cache/numa information into guest
    /// Generate fdt information about cache/numa
    fn get_fdt_numa_info(&self, enable_cache: bool) -> FdtNumaInfo {
        // Get numa_regions from vm_config.
        let mut numa_regions = self.vm_config().numa_regions.clone();
        // Generate vcpu(index)->pcpu(value) maps.
        // Currently only assure mappings from vcpu to numa.
        // Vcpu to pcpu mappings is a future feature.
        // When numa_regions is empty, there is no meaning to passthrough
        // cache information into guest, so just skip it.
        let cpu_maps = if enable_cache {
            numa_regions.sort_by_key(|numa_info| numa_info.host_numa_node_id);
            // Unwrap pcpu_ids here is safe because we have check it in get_fdt_vm_info.
            Some(
                numa_regions
                    .iter()
                    .flat_map(|numa_info| {
                        numa_info
                            .pcpu_ids
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|val| *val as u8)
                    })
                    .collect::<Vec<u8>>(),
            )
        } else {
            None
        };

        // generate vectors of numa id for each memory region
        let numa_nodes = self.address_space.get_numa_nodes();
        let memory_numa_id = numa_nodes
            .iter()
            .flat_map(|(numa_id, numa_node)| {
                numa_node
                    .region_infos()
                    .iter()
                    .map(|_| *numa_id)
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<u32>>();
        // generate vectors of numa id for each cpu
        let cpu_numa_id = numa_nodes
            .iter()
            .flat_map(|(numa_id, numa_node)| {
                numa_node
                    .vcpu_ids()
                    .iter()
                    .map(|_| *numa_id)
                    .collect::<Vec<u32>>()
            })
            .collect::<Vec<u32>>();
        // generate vector that saves shared cpu count on the same numa node
        let mut cpu_l3_cache_map = vec![0u32; cpu_numa_id.len()];
        for (_, numa_info) in numa_nodes.iter() {
            let count = numa_info.vcpu_ids().iter().min().unwrap();
            for id in numa_info.vcpu_ids().iter() {
                cpu_l3_cache_map[*id as usize] = *count;
            }
        }
        FdtNumaInfo::new(cpu_maps, memory_numa_id, cpu_numa_id, cpu_l3_cache_map)
    }

    /// Generate fdt information about devices
    fn get_fdt_device_info(&self) -> FdtDeviceInfo<MMIODeviceInfo> {
        FdtDeviceInfo::new(
            self.device_manager().get_mmio_device_info(),
            self.get_irqchip(),
        )
    }

    /// Execute system architecture specific configurations.
    ///
    /// 1) set guest kernel boot parameters
    /// 2) setup FDT data structs.
    pub fn configure_system_arch(
        &self,
        vm_memory: &GuestMemoryImpl,
        cmdline: &Cmdline,
        initrd: Option<InitrdConfig>,
    ) -> std::result::Result<(), StartMicroVmError> {
        let vcpu_manager = self.vcpu_manager().map_err(StartMicroVmError::Vcpu)?;
        let cmdline_cstring = cmdline
            .as_cstring()
            .map_err(StartMicroVmError::ProcessCommandline)?;
        let fdt_vm_info = self.get_fdt_vm_info(
            vm_memory,
            cmdline_cstring
                .to_str()
                .map_err(|_| StartMicroVmError::ProcessCommandline(CmdlineError::InvalidAscii))?,
            initrd.as_ref(),
            &vcpu_manager,
        );
        let fdt_numa_info = self.get_fdt_numa_info(fdt_vm_info.get_cache_passthrough_enabled());
        let fdt_device_info = self.get_fdt_device_info();

        dbs_boot::fdt::create_fdt(fdt_vm_info, fdt_numa_info, fdt_device_info)
            .map(|_| ())
            .map_err(|e| StartMicroVmError::ConfigureSystem(Error::BootSystem(e)))
    }

    pub(crate) fn register_events(
        &mut self,
        event_mgr: &mut EventManager,
    ) -> std::result::Result<(), StartMicroVmError> {
        let reset_evt = self.get_reset_eventfd().ok_or(StartMicroVmError::EventFd)?;
        event_mgr
            .register_exit_eventfd(reset_evt)
            .map_err(|_| StartMicroVmError::RegisterEvent)?;

        Ok(())
    }
}
